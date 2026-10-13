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
v2 brain (brain.version = 2; with version 1 or no version every line above holds bit for bit): see V2_DOC below.
Usage: uv run python scripts/mission.py            (self-check: 10 easy meadow seeds, ring and vote; v2 smoke check)
       uv run python scripts/mission.py --time     (time a batch of 50 seeds x 300 s, print the extrapolation)
"""
import json, math, os, sys, time, types
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(HERE).parent
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
from home import HomeMemory, F_HOME, F_TRACK  # noqa: E402
import mb_front  # noqa: E402
import arbiters_v2  # noqa: E402
import search  # noqa: E402
from scenario import MODULES  # noqa: E402

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
SPIRAL_B_NOTIP = 12.0          # no operator tip (brain.tip = null): spiral from home, wider pitch
INSTR_BEAR_SD, INSTR_DIST_SD = math.radians(8.0), 0.1
HOLD_DEG, CROSS_RATIO, REL_CAP = 20.0, 1.5, 20.0
OCC_NEAR = 10.0
N_KC, K_IN, KC_FRAC, MB_SEED = 20000, 10, 0.01, 0
V2_DOC = """v2 brain (fixed before any v2 mission data; the paper's v2 section):
  * Home pad: a 1 x 1 m patterned mat at home (world.home_pad), not an object slot.
  * home: take-off learning flight (T_HLF s): climb for HLF_CLIMB s, then an outward loop around the pad referenced to
    the PI (radius 0.5 -> 2.2 m, ~0.3 m lag, inside the 3 m obstacle-free disc, 0.75 turn, climbing 1 -> 3 m; velocity = feed-forward
    + 1.5/s position error, <= 2 m/s), flown forwards because the Stone PI integrates forward motion only (the panoramic
    eye sees the pad from any heading); every 10 Hz look of the
    pad is stored in the home memory (brain/home.py; Rng stream 21 pixel noise). On return the home memory scores the
    pad and every visible object once the PI says home is within R_FINAL; a recognised pad (familiarity >= F_HOME) replaces
    the path-integrator home vector by the visual vector.
    Without the home module the take-off lasts T_TAKEOFF s as in v1.
  * landing: final approach when the approach vector (visual if the pad is locked, else path integrator) is shorter
    than R_FINAL m (or the pad is locked). Inside D_SERVO m the drone servoes its velocity onto the pad (gain K_SERVO,
    <= V_FINAL), facing the approach vector until 0.5 m out, then holding its heading. With the home module and no lock when the PI says "here" (< 1 m), it searches the
    pad on an outward spiral (PAD_SPIRAL_B m pitch) for up to PAD_SEARCH_S s, then lands on the PI estimate. Descent
    starts within D_DESC m at horizontal speed < V_HSTILL; the altitude set-point then falls at V_DESC m/s (servo <= V_DESC_H); touchdown
    = first ground contact; motors are cut (disarm). Without the landing module the mission ends as in v1 when the
    approach vector is < 1 m (hovering).
  * heading: gyro z (body yaw rate averaged over the last brain tick + per-seed bias N(0, 0.5 deg/s) + N(0, 0.1 deg/s)
    per tick, Rng stream 20) and the sun compass (the v1 TL->CL1->TB1 decode of true yaw + N(0, 2 deg) + false-sun
    offset; compass_half = its CL1 column mask) go to the heading estimator (brain/heading.py; HeadingRing by default,
    module_params.heading_estimator = "kf" | "cf" for the comparison arms; params results/kt5_tuned_params.json); its
    estimate is the path integrator's compass input (scripts/kt5_heading.py wiring).
  * front / inspect / gated_arbiter: brain/mb_front.py, brain/arbiters_v2.py when present.
  * adaptive_search (post-study, optional, not in the Study 2 default list): brain/search.py (size estimate from the
    learning flight -> search altitude and spiral pitch, attention window and self-calibrated evidence window, inspection
    dips). Off = Study 2 bit for bit.
  Run-record fields added (both versions): landing_error_m (horizontal distance from the pad centre at touchdown, or
  at the hover end without the landing module), landed_on_pad (<= PAD_R), touchdown_speed_mps (NaN without touchdown),
  success_full = success_target & landed_on_pad & no collision."""
T_HLF, HLF_CLIMB = 8.0, 1.0
ALT_SEARCH = 3.5               # m, v2 search / return altitude (train-seed choice, see run_v2.py calib_alt)
ALT_INSPECT_MIN, INSPECT_NEAR = 0.8, 8.0   # m: inspection altitude = max(0.8, object height) within 8 m of a candidate
R_FINAL, D_SERVO, K_SERVO, V_FINAL = 6.0, 2.0, 0.8, 0.6
D_DESC, V_HSTILL, V_DESC, V_DESC_H = 0.25, 0.25, 0.3, 0.2
PAD_SEARCH_S, PAD_SPIRAL_B, PAD_R = 40.0, 3.0, 0.5
GYRO_BIAS_SD, GYRO_NOISE_SD = math.radians(0.5), math.radians(0.1)   # rad/s, as scripts/kt5_heading.py
HEADING_DEV = {"w_sun": 0.5, "gate_deg": 25.0, "remap_delay_s": 3.0, "remap_tau_s": 5.0}   # dev fallback (heading.py self-check values)
F0_FRONT, F1_FRONT = 0.330, 0.371       # familiarity -> evidence with the front end: results/v2_calib.json (train 100-139)
F_HOME_FRONT, F_TRACK_FRONT = 0.44, 0.22   # home-memory thresholds with the front end: results/v2_calib.json
INSPECT_WIN, INSPECT_R, INSPECT_SEP, ORBIT_V = 30, 4.0, 20.0, 0.5
V2_ARB = {"ring": "ring_gated", "ring_static": "ring_static_vg", "vote": "vote_vg", "sum": "sum_vg",
          "accumulator": "accumulator_vg", "state_machine": "state_machine"}
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


_WN = {}


def count_matrix_n(n_in):
    """[n_in, N_KC] counts for another input size (v2 front end: MushroomBody(n_in, seed=MB_SEED) wiring)."""
    if n_in not in _WN:
        idx = mbmod.wiring(n_in, N_KC, K_IN, MB_SEED)
        _WN[n_in] = np.zeros((n_in, N_KC), np.float32)
        np.add.at(_WN[n_in], (idx, np.arange(N_KC)[:, None]), 1.0)
    return _WN[n_in]


def front_ranks(rgb):
    """v2 front end: RGB windows [..., H, W, 3] -> integer per-channel ranks 1..36 (= mb_front.front x 36). The pooled
    features are quantised to 1e-9 before the stable per-channel rank, as ranks() quantises grey looks: uniform cells
    (ground, sky) tie exactly in numpy but differ in the last bit in JS, which would break ties differently."""
    g2 = mb_front.CFG["grid"] ** 2
    f = mb_front.features(rgb, **mb_front.CFG)
    q = np.floor(f.reshape(*f.shape[:-1], -1, g2) * 1e9 + 0.5)
    o = np.argsort(q, axis=-1, kind="stable")
    r = np.empty_like(o)
    np.put_along_axis(r, o, np.broadcast_to(np.arange(1, g2 + 1), o.shape), -1)
    return r.reshape(f.shape).astype(np.int64)


def kc_active(rk):
    """rank vectors [N, n] -> active KCs [N, N_KC]. Integer form of mb.MushroomBody.active on mb.equalise inputs
    (KC drive = sum of the ranks of its 10 PNs; the L2 normalisation is a common factor), exact in Python and JS."""
    d = rk.astype(np.float32) @ (count_matrix() if rk.shape[-1] == eye.LH * eye.LW else count_matrix_n(rk.shape[-1]))
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
            assert _v2key(s["brain"]) == _v2key(br), "a batch shares one brain version and module set"
            assert len(s["decoys"]) + 1 <= K_MAX
        self.version = br.get("version", 1)
        self.v2 = self.version == 2
        if self.v2:
            assert all(s["target"]["kind"] != "pad" and all(d["kind"] != "pad" for d in s["decoys"]) for s in self.sc), \
                "brain.version 2 searches for 3D objects: no pad targets or pad decoys"
        self.mods = set(br.get("modules", MODULES)) if self.v2 else set()
        self.mp = br.get("module_params", {})
        self.front = "front" in self.mods
        self.look_kw = {"half_m": mb_front.CFG["half_m"], "rgb": True} if self.front else {}
        self.look_shape = (eye.LH * eye.SS, eye.LW * eye.SS, 3) if self.front else (eye.LH * eye.LW,)
        self.look_n = int(np.prod(self.look_shape))
        self.f0, self.f1 = (F0_FRONT, F1_FRONT) if self.front else (F0, F1)
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
        tips = [s["brain"].get("tip", "v1") for s in self.sc]          # absent = the v1 instruction; None = no tip
        bsd = np.array([INSTR_BEAR_SD if t == "v1" or t is None else math.radians(t["bearing_sd_deg"]) for t in tips])
        dsd = np.array([INSTR_DIST_SD if t == "v1" or t is None else float(t["dist_sd"]) for t in tips])
        ang = np.arctan2(rel[:, 1], rel[:, 0]) + bsd * nb
        dist = np.hypot(rel[:, 0], rel[:, 1]) * (1 + dsd * nd)
        self.goal = np.stack([dist * np.cos(ang), dist * np.sin(ang)], 1)        # PI frame (home = origin)
        notip = np.array([t is None for t in tips])
        self.goal = np.where(notip[:, None], 0.0, self.goal)          # no tip: the spiral starts around home
        self.spb = np.where(notip, SPIRAL_B_NOTIP, SPIRAL_B)
        self.ad = "adaptive_search" in self.mods
        if self.ad:
            self._init_adapt(br)
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
            if self.front:
                L = eye.learning_looks(self.S, lf, half_m=self.half_m[:, None] if self.ad else mb_front.CFG["half_m"], rgb=True)[rows]
                act = self.code(L.reshape(-1, *L.shape[2:])).reshape(len(rows), L.shape[1], N_KC)
            else:
                L = eye.learning_looks(self.S, lf)[rows]                               # [b, V, 192]
                act = kc_active(ranks(L.reshape(-1, L.shape[-1]))).reshape(len(rows), L.shape[1], N_KC)
            self.mbw[rows] &= ~act.any(1)
        if self.ad:
            self._calibrate()
        params = dict(br["params"])
        if br["arbiter"] == "ring":
            params["ring_gain"] = br["ring_gain"]
        self.arb_name = br["arbiter"]
        if "gated_arbiter" in self.mods:
            self.arb = arbiters_v2.make(V2_ARB[br["arbiter"]], params, B)
        else:
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
        self.landed = np.zeros(B, bool); self.land_err = np.full(B, np.nan)   # bookkeeping (both versions)
        self.hd_on = "heading" in self.mods
        self.t_takeoff = T_HLF if "home" in self.mods else T_TAKEOFF
        if self.v2:
            self._init_v2()

    def _init_v2(self):
        B = self.B
        self.PS = eye.Scene([{**w, "objects": [w["home_pad"]]} for w in self.worlds])     # the pad as a one-slot scene
        self.r_gyro, self.r_pad = Rng(self.seeds, 20), Rng(self.seeds, 21)
        self.gyro_bias = GYRO_BIAS_SD * self.r_gyro.normal()
        self.gyro_acc = np.zeros(B)
        self.hm = HomeMemory(B, self.code, N_KC, *((F_HOME_FRONT, F_TRACK_FRONT) if self.front else (F_HOME, F_TRACK)))
        self.obj_g = np.zeros((B, self.K) + self.look_shape)     # last noisy looks of the object slots
        if self.hd_on:                    # estimator: module_params.heading_estimator ring (default) | kf | cf
            import heading
            est = self.mp.get("heading_estimator", "ring")
            hp = self.mp.get("heading")
            if hp is None:
                f = ROOT_DIR / "results/kt5_tuned_params.json"
                assert f.exists() or est == "ring", "frozen heading params missing"
                hp = json.loads(f.read_text())["estimators"][est]["params"] if f.exists() else HEADING_DEV
            self.hd = heading.CLASSES[est](B, hp)
            self.sun_pi = PathIntegrator(B)                       # the v1 sun compass (TL -> CL1 -> TB1), zero velocity
        if "inspect" in self.mods:
            self.ins_az = np.zeros((B, self.K, INSPECT_WIN)); self.ins_ev = np.zeros((B, self.K, INSPECT_WIN))
            self.ins_ok = np.zeros((B, self.K, INSPECT_WIN), bool)
        self.pad_vis, self.home_fam, self.seen = np.zeros(B, bool), np.zeros(B), np.zeros(B, bool)
        self.hstage = np.zeros(B, int)                          # 0 cruise home, 1 final approach, 2 descent
        self.ap = np.zeros((B, 2))                              # approach vector (compass frame, m)
        self.pad_t0 = np.full(B, np.nan); self.pspiral = np.zeros(B); self.searching = np.zeros(B, bool)
        self.alt_sp = np.full(B, ALT)
        self.touch = np.zeros(B, bool); self.touch_speed = np.full(B, np.nan); self.touch_pos = np.full((B, 2), np.nan)
        self.lf_phi0 = wrap(self.heading_est + np.pi)                # learning-flight arcs start behind the drone

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
        if not self.hd_on:              # v1 (and v2 without the heading module): compass_half silences PI compass columns
            self.pi.col_mask[t >= self.t_comp] = np.tile(np.r_[np.zeros(4), np.ones(4)], 2)
        # compass (stream 2) and odometry (stream 3)
        theta = yaw + SIGMA_COMPASS * self.r_comp.normal() + np.where(sun_on, self.false_sun, 0.0)
        c, s_ = np.cos(yaw), np.sin(yaw)
        vb = np.stack([c * q.vel[:, 0] + s_ * q.vel[:, 1], -s_ * q.vel[:, 0] + c * q.vel[:, 1]], 1)
        n1, n2 = self.r_speed.normal(), self.r_speed.normal()
        scale = np.where(t >= self.t_speed, self.speed_scale, 1.0)
        vmeas = scale[:, None] * vb + SIGMA_SPEED * np.stack([n1, n2], 1)
        if self.hd_on:
            theta = self._heading(theta, t >= self.t_comp)
        self.heading_est = self.pi.step(theta, vmeas, DT_B)
        hv = self.pi.home_vector()
        dead = np.where(cols_on[:, None], self.dead_cols, False)
        # eye at 10 Hz
        if self.tick % EYE_EVERY == 0:
            self._eye(pos, yaw, vmeas, dead, live)
            if "home" in self.mods:
                self._eye_home(pos, yaw, dead, hv)
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
            r = self.spb * self.spiral / (2 * np.pi)
            self.spiral = np.where(near, self.spiral + np.minimum(0.5, 1.0 / np.maximum(r, 1.0)), self.spiral)
        tgt = np.where((self.spiral >= 0)[:, None], self._spiral_wp(), self.goal)
        search_b = np.arctan2(tgt[:, 1] - p_hat[:, 1], tgt[:, 0] - p_hat[:, 0])
        homing = self.phase == 3
        seeking = (self.phase == 1) | (self.phase == 2)
        obs = {"objects": {"bearing": bearing, "strength": np.where(seeking[:, None], self.strength, 0.0),
                           "visible": self.vis & seeking[:, None]},
               "home": {"bearing": np.arctan2(hv[:, 1], hv[:, 0]) if not self.v2 else self._approach(hv),
                        "strength": homing.astype(float)},
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
        self.phase = np.where((self.phase == 0) & (t >= self.t_takeoff), 1, self.phase)
        seeking = (self.phase == 1) | (self.phase == 2)
        on_obj = (self.vis & (self.strength > 0.2) & (np.abs(wrap(bearing - cmd[:, None])) < math.radians(30))).any(1)
        self.phase = np.where(seeking, np.where((res["commit"] > 0.5) & on_obj, 2, 1), self.phase)
        cand = self.vis & (self.rh < STOP_R) & (self.ev > STOP_THR) & seeking[:, None]
        if "inspect" in self.mods:              # inspection: familiar from >= 2 viewpoints >= INSPECT_SEP deg apart
            cand &= mb_front.inspect_ok(self.ins_az, self.ins_ev, self.ins_ok, STOP_THR, INSPECT_SEP)
        st_t, st_d = (cand & self.is_t).any(1), (cand & self.is_d).any(1)
        st_d &= ~st_t
        self.out["success_target"] |= st_t
        self.out["wrong_object_stop"] |= st_d
        self.phase = np.where(st_t | st_d, 3, self.phase)
        landed = (self.phase == 3) & (np.hypot(hv[:, 0], hv[:, 1]) < 1.0) if not self.v2 else self._landing(pos)
        hd_true = np.hypot(pos[:, 0] - self.home[:, 0], pos[:, 1] - self.home[:, 1])
        new_land = landed & live & ~self.landed
        self.land_err = np.where(new_land, hd_true, self.land_err)
        self.landed |= new_land
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
        dh = np.hypot(hv[:, 0], hv[:, 1]) if not self.v2 else np.hypot(self.ap[:, 0], self.ap[:, 1])
        v = np.where(self.phase == 3, np.minimum(v, np.maximum(0.4, dh / 2)), v)
        v = np.where((self.phase == 0) | ~live, 0.0, v)
        sp = {"vel_xy": v[:, None] * np.stack([np.cos(yaw_sp), np.sin(yaw_sp)], 1),
              "alt": np.where(live, ALT, 0.0), "yaw": np.where(self.phase == 0, yaw, yaw_sp)}
        if self.v2:
            sp = self._sp_v2(sp, yaw, hv, pos, live)
        if self.log_rows and self.tick % EYE_EVERY == 0:
            self._log(res, hv, dead, sun_on, yaw, sp["alt"])
        if self.v2:
            self.gyro_acc[:] = 0.0
        for _ in range(SUB):
            for rows_g, d in self.wind_groups:
                self.wind[rows_g] = d.step(DT_B / SUB)
            m = self.ctrl(q, sp, DT_B / SUB)
            if self.v2:
                m = np.where(self.touch[:, None], 0.0, m)                 # disarmed after touchdown
                vprev = q.vel.copy()
            q.step(m, self.wind, DT_B / SUB)
            if self.v2:
                self.gyro_acc += q.omega[:, 2]
                td = (self.hstage == 2) & ~self.touch & (q.pos[:, 2] <= 0.0) & (self.phase < 4)
                self.touch_speed = np.where(td, np.linalg.norm(vprev, axis=1), self.touch_speed)
                self.touch_pos = np.where(td[:, None], q.pos[:, :2], self.touch_pos)
                self.touch |= td
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
        r = self.spb * ph / (2 * np.pi)
        a = self.spiral0 + ph
        return self.goal + np.stack([r * np.cos(a), r * np.sin(a)], 1)

    def _eye(self, pos, yaw, vmeas, dead, live):
        S, B, K, t = self.S, self.B, self.K, self.t
        rows = np.arange(B)
        eye_pos = pos.copy()
        occ = ((t >= self.occ_on) & (t < self.occ_on + self.occ_dur))[:, None] & self.is_t
        vis, bear, rh = eye.visibility(S, eye_pos, yaw, t, dead=dead, occluded=occ)
        noise = eye.normals(self.r_eye, K_MAX * self.look_n).reshape(B, K_MAX, -1)[:, :K]
        fam = np.zeros((B, K))
        idx = np.argwhere(vis)
        if len(idx):
            grey, _ = eye.looks(eye.sub_scene(S, idx[:, 0], idx[:, 1]), eye_pos[idx[:, 0]], t, **self._look_kw(idx[:, 0]))
            g = grey[:, 0] + eye.NOISE * noise[idx[:, 0], idx[:, 1]].reshape(grey[:, 0].shape)
            act = kc_active(ranks(g)) if not self.front else self.code(g)
            fam[idx[:, 0], idx[:, 1]] = 1.0 - (act & self.mbw[idx[:, 0]]).sum(-1) / act.sum(-1)
            if self.v2:
                self.obj_g[idx[:, 0], idx[:, 1]] = g
        self.fam = fam
        self.vis, self.rh = vis, rh
        self.bear_rel = wrap(bear - yaw[:, None])
        f0, f1 = (self.f0[:, None], self.f1[:, None]) if self.ad else (self.f0, self.f1)
        self.strength = np.where(vis, np.clip((fam - f0) / (f1 - f0), 0.0, 1.0), 0.0)
        self.hist = np.roll(self.hist, 1, -1)
        self.hist[..., 0] = self.strength
        self.ev = self.hist.mean(-1)
        if "inspect" in self.mods:              # view azimuth (object -> eye, compass frame), 1-s evidence, valid
            for a, v in ((self.ins_az, wrap(self.bear_rel + self.heading_est[:, None] + np.pi)), (self.ins_ev, self.ev),
                         (self.ins_ok, vis & (rh < INSPECT_R))):
                a[...] = np.roll(a, 1, -1)
                a[..., 0] = v
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

    # ---------------------------------------------------------------- v2 brain modules
    def _init_adapt(self, br):
        """adaptive_search: size estimate from the learning flight (prior without one), search geometry, attention window."""
        B, S = self.B, self.S
        rows, tk = np.arange(B), np.argmax(S.role, 1)
        T = S.obj[rows, tk]
        self.size_d, self.size_h = np.full(B, search.D_PRIOR), np.full(B, search.H_PRIOR)
        self.lf_view = np.zeros((B, 4))                     # first / last learning pose: horizontal range, altitude
        for lf in sorted({s["brain"]["learning_flight_s"] for s in self.sc}):
            r = np.array([i for i, s in enumerate(self.sc) if s["brain"]["learning_flight_s"] == lf])
            P = eye.learning_poses(S, lf)[r]
            dx, dy = P[..., 0] - T[r, 0][:, None], P[..., 1] - T[r, 1][:, None]
            d = np.sqrt(dx * dx + dy * dy)
            self.lf_view[r] = np.stack([d[:, 0], P[:, 0, 2], d[:, -1], P[:, -1, 2]], 1)
            if P.shape[1] > 1:
                self.size_d[r], self.size_h[r] = search.size_estimate(P, T[r, 0], T[r, 1], T[r, 6], eye.obj_diag(S)[r, tk[r]])
        self.h_search, self.spb, self.r_dip = search.geometry(self.size_d, self.size_h, self.spb, ALT_SEARCH)
        self.half_m = search.half_m(self.size_d)
        self.inspecting = np.zeros(B, bool)

    def _calibrate(self):
        """adaptive_search: evidence window from unstored look-back views of the target (Rng stream 22 pixel noise)."""
        B, S = self.B, self.S
        rows, tk = np.arange(B), np.argmax(S.role, 1)
        T = S.obj[rows, tk]
        C = search.cal_poses(T[:, 0], T[:, 1], self.size_d, self.h_search)
        noise = eye.normals(Rng(self.seeds, 22), C.shape[1] * self.look_n).reshape(B, C.shape[1], -1)
        fam = np.zeros((B, C.shape[1]))
        for v in range(C.shape[1]):
            g, _ = eye.looks(eye.sub_scene(S, rows, tk), C[:, v], 1e9, **self._look_kw(rows))
            g = g[:, 0] + eye.NOISE * noise[:, v].reshape(g[:, 0].shape)
            act = self.code(g)
            fam[:, v] = 1.0 - (act & self.mbw).sum(-1) / act.sum(-1)
        self.cal_fam = fam
        self.f0, self.f1 = search.evidence_window(fam)

    def _look_kw(self, rows):
        """Object-look options for these batch rows (adaptive_search with the front end: per-row attention window)."""
        return {**self.look_kw, "half_m": self.half_m[rows][:, None]} if self.ad and self.front else self.look_kw

    def code(self, looks):
        """Looks -> active KCs (grey ranks, or the front end's per-channel ranks)."""
        return kc_active(front_ranks(np.clip(looks, 0.0, 1.0))) if self.front else kc_active(ranks(looks))

    def _heading(self, theta, comp_on):
        """Heading module: mean body yaw rate over the last tick + bias + noise (stream 20), and the v1 sun compass
        (compass_half silences its CL1 columns). -> fused heading [B] (the PI's compass input)."""
        gyro = self.gyro_acc / SUB + self.gyro_bias + GYRO_NOISE_SD * self.r_gyro.normal()
        self.sun_pi.col_mask[:] = 1.0
        self.sun_pi.col_mask[comp_on] = np.tile(np.r_[np.zeros(4), np.ones(4)], 2)
        sun = self.sun_pi.step(theta, np.zeros((self.B, 2)), DT_B)
        return self.hd.step(gyro, sun, np.ones(self.B, bool), DT_B)

    def _eye_home(self, pos, yaw, dead, hv):
        """10 Hz: the pad's look (learning flight: store it; homing: score the pad and the object slots)."""
        B, t = self.B, self.t
        lf = (self.phase == 0) & (t >= HLF_CLIMB)
        hom = (self.phase == 3) & ((np.hypot(hv[:, 0], hv[:, 1]) < R_FINAL) | self.hm.locked)   # end of the home vector
        pv, pb, prh = eye.visibility(self.PS, pos, yaw, t, dead=dead)
        pv = pv[:, 0] & (lf | hom)
        noise = eye.normals(self.r_pad, self.look_n)
        g = np.zeros((B,) + self.look_shape)
        ix = np.flatnonzero(pv)
        if len(ix):
            gl, _ = eye.looks(eye.sub_scene(self.PS, ix, np.zeros(len(ix), int)), pos[ix], t, **self.look_kw)
            g[ix] = gl[:, 0] + eye.NOISE * noise[ix].reshape(gl[:, 0].shape)
        self.pad_vis = pv
        lr = ix[lf[ix]]
        self.hm.learn(lr, g[lr])
        looks = np.concatenate([g[:, None], self.obj_g], 1)
        vis = np.concatenate([pv[:, None], self.vis & self.S.kvalid], 1) & hom[:, None]
        bc = wrap(np.concatenate([(pb[:, 0] - yaw)[:, None], self.bear_rel], 1) + self.heading_est[:, None])
        rh = np.concatenate([prh, self.rh], 1)
        _, self.home_fam, _, self.seen = self.hm.steer(looks, vis, bc, rh, hv)

    def _approach(self, hv):
        """Approach vector (compass frame): visual (home memory) once the pad is locked, else the PI home vector; during
        the pad search, the next point of a spiral around the PI home. -> its bearing [B]."""
        ap = np.where(self.hm.locked[:, None], hv + self.hm.corr, hv) if "home" in self.mods else hv.copy()
        if self.searching.any():
            for _ in range(100):                                      # keep the spiral point >= 1.5 m ahead
                wp = self._pad_wp() + hv
                near = self.searching & (np.hypot(wp[:, 0], wp[:, 1]) < 1.5)
                if not near.any():
                    break
                r = PAD_SPIRAL_B * self.pspiral / (2 * np.pi)
                self.pspiral = np.where(near, self.pspiral + np.minimum(0.5, 1.0 / np.maximum(r, 1.0)), self.pspiral)
            ap = np.where(self.searching[:, None], self._pad_wp() + hv, ap)
        self.ap = ap
        return np.arctan2(ap[:, 1], ap[:, 0])

    def _pad_wp(self):
        r = PAD_SPIRAL_B * self.pspiral / (2 * np.pi)
        return np.stack([r * np.cos(self.pspiral), r * np.sin(self.pspiral)], 1)

    def _landing(self, pos):
        """Final approach / pad search / descent stages. -> landed [B] (touchdown; or the v1-style hover end)."""
        home = self.phase == 3
        d = np.hypot(self.ap[:, 0], self.ap[:, 1])
        if "landing" not in self.mods:
            return home & (d < 1.0)
        locked = self.hm.locked & ("home" in self.mods)
        self.hstage = np.where(home & (self.hstage == 0) & ((d < R_FINAL) | locked), 1, self.hstage)
        if "home" in self.mods:
            start = home & (self.hstage == 1) & ~locked & (d < 1.0) & np.isnan(self.pad_t0)
            self.pad_t0 = np.where(start, self.t, self.pad_t0)
            self.searching = home & (self.hstage == 1) & ~locked & (self.t - np.nan_to_num(self.pad_t0, nan=np.inf) < PAD_SEARCH_S)
        hs = np.hypot(self.quad.vel[:, 0], self.quad.vel[:, 1])
        ready = home & (self.hstage == 1) & ~self.searching & (d < D_DESC) & (hs < V_HSTILL)
        self.alt_sp = np.where(ready, pos[:, 2], self.alt_sp)
        self.hstage = np.where(ready, 2, self.hstage)
        return home & self.touch

    def _sp_v2(self, sp, yaw, hv, pos, live):
        """Set-point overrides: the take-off learning flight, the velocity servo onto the pad, the descent."""
        vel, alt, yw = sp["vel_xy"].copy(), np.array(sp["alt"], float), np.array(sp["yaw"], float)
        # altitude: search/return high, drop to the object's height near a candidate (inspection)
        cand = self.vis & (self.rh < INSPECT_NEAR) & (self.strength > 0.3) & ((self.phase == 1) | (self.phase == 2))[:, None]
        k = np.argmin(np.where(cand, self.rh, np.inf), 1)
        h_obj = np.maximum(ALT_INSPECT_MIN, self.S.obj[np.arange(self.B), k, 6])
        alt = np.where(live, np.where(cand.any(1), h_obj, ALT_SEARCH), 0.0)
        if self.ad:                             # adaptive search altitude and inspection dips (hysteresis DIP_ON / DIP_OFF)
            seek = (self.phase == 1) | (self.phase == 2)
            thr = np.where(self.inspecting, search.DIP_OFF, search.DIP_ON)[:, None]
            cand = self.vis & (self.ev >= thr) & (self.rh < self.r_dip[:, None]) & seek[:, None]
            self.inspecting = cand.any(1)
            dip = np.clip(search.view_alt(np.where(cand, self.rh, np.inf).min(1), *self.lf_view.T), np.maximum(search.ALT_MIN, self.size_h), self.h_search)
            alt = np.where(live, np.where(self.inspecting, dip, np.where(seek, self.h_search, ALT_SEARCH)), 0.0)
        rot = yaw - self.heading_est                                   # compass frame -> airframe
        c, s_ = np.cos(rot), np.sin(rot)
        to_air = lambda v: np.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1]], 1)  # noqa: E731
        if "home" in self.mods:
            tau = self.t
            lf = self.phase == 0
            climb, arc = lf & (tau < HLF_CLIMB), lf & (tau >= HLF_CLIMB)
            # PI-referenced outward loop around the pad inside the obstacle-free disc, flown forwards (the Stone PI integrates
            # forward motion only; the panoramic eye sees the pad from any heading): r 0.5 -> 2.2 m, 0.75 turn
            u = (tau - HLF_CLIMB) / (T_HLF - HLF_CLIMB)
            r, dr = 0.5 + 1.7 * u, 1.7 / (T_HLF - HLF_CLIMB)
            ph, dph = self.lf_phi0 + 1.5 * np.pi * u, 1.5 * np.pi / (T_HLF - HLF_CLIMB)
            e = np.stack([np.cos(ph), np.sin(ph)], 1)
            vc = dr * e + r * dph * np.stack([-e[:, 1], e[:, 0]], 1) + 1.5 * (r * e + hv)       # feed-forward + P on -hv
            vc = vc * np.minimum(1.0, 2.0 / np.maximum(np.hypot(vc[:, 0], vc[:, 1]), 1e-9))[:, None]
            vel = np.where(arc[:, None], to_air(vc), vel)
            alt = np.where(climb, 1.0, np.where(arc, 1.0 + (ALT_SEARCH - 1.0) * u, alt))
            yw = np.where(arc, wrap(np.arctan2(vc[:, 1], vc[:, 0]) + rot), yw)
        if "inspect" in self.mods:              # sideways drift near a candidate, so the views differ
            orb = (((self.phase == 1) | (self.phase == 2)) & (self.vis & (self.rh < INSPECT_R) & (self.ev > 0.3)).any(1))
            vel = vel + np.where(orb[:, None], ORBIT_V * np.stack([-np.sin(yw), np.cos(yw)], 1), 0.0)
        if "landing" in self.mods:
            d = np.hypot(self.ap[:, 0], self.ap[:, 1])
            servo = (self.phase == 3) & (((self.hstage == 1) & (d < D_SERVO) & ~self.searching) | (self.hstage == 2))
            vc = K_SERVO * self.ap
            vmax = np.where(self.hstage == 2, V_DESC_H, V_FINAL)              # slower sideways while descending
            vc = vc * np.minimum(1.0, vmax / np.maximum(np.hypot(vc[:, 0], vc[:, 1]), 1e-9))[:, None]
            vel = np.where(servo[:, None], to_air(vc), vel)
            yw = np.where(servo, np.where(d > 0.5, wrap(np.arctan2(self.ap[:, 1], self.ap[:, 0]) + rot), yaw), yw)   # face the motion (PI)
            desc = (self.hstage == 2) & live
            self.alt_sp = np.where(desc, np.maximum(self.alt_sp - V_DESC * DT_B, -0.2), self.alt_sp)
            alt = np.where(desc, self.alt_sp, alt)
        return {"vel_xy": vel, "alt": alt, "yaw": yw}

    def _log(self, res, hv, dead, sun_on, yaw, alt_sp):
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
            vel = q.vel[b]
            stages = ["cruise", "final", "descent"]
            self.frames[b][-1]["hud"] = {
                "alt_m": float(q.pos[b, 2]), "alt_sp_m": float(np.broadcast_to(alt_sp, (self.B,))[b]), "vz_mps": float(vel[2]),
                "ground_speed_mps": float(np.hypot(vel[0], vel[1])), "air_speed_mps": float(np.linalg.norm(vel - self.wind[b])),
                "heading_true_deg": math.degrees(yaw[b]), "heading_est_deg": math.degrees(self.heading_est[b]),
                "heading_cmd_deg": math.degrees(res["heading_cmd"][b]), "roll_deg": math.degrees(eul[b, 0]), "pitch_deg": math.degrees(eul[b, 1]),
                "home_dist_est_m": float(np.hypot(*(hv[b] if not self.v2 else self.ap[b]))),
                "home_dist_true_m": float(np.hypot(*(self.home[b, :2] - q.pos[b, :2]))), "energy_so_far": float(self.energy[b]),
                "phase": names[self.phase[b]], "version": self.version,
                "heading_conf": float(self.hd.conf[b]) if self.hd_on else None,
                "home_familiarity": float(self.home_fam[b]) if "home" in self.mods else None,
                "landing_stage": (("touchdown" if self.touch[b] else stages[self.hstage[b]]) if "landing" in self.mods else None)}
            if self.ad:
                self.frames[b][-1]["hud"] |= {"search_alt_m": float(self.h_search[b]), "size_est_m": float(self.size_d[b]),
                                              "inspecting": bool(self.inspecting[b])}
            if self.v2:
                self.frames[b][-1]["home"] = {
                    "stage": int(self.hstage[b]), "pad_visible": bool(self.pad_vis[b]), "familiarity": float(self.home_fam[b]),
                    "locked": bool(self.hm.locked[b]), "seen": bool(self.seen[b]), "searching": bool(self.searching[b]),
                    "approach": self.ap[b].tolist(), "views": int(self.hm.n_views[b]), "touchdown": bool(self.touch[b])}

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
                "t_release_decoy_s": np.where(self.has_dec, t_rel, np.nan),
                "landing_error_m": self.land_err.copy(), "landed_on_pad": self.landed & (self.land_err <= PAD_R),
                "touchdown_speed_mps": self.touch_speed.copy() if self.v2 else np.full(self.B, np.nan),
                "success_full": self.out["success_target"] & self.landed & (self.land_err <= PAD_R) & (self.collisions == 0),
                **({"size_est_m": self.size_d.copy(), "search_alt_m": self.h_search.copy()} if self.ad else {})}

    def outcome_row(self, b):
        o = self.finish()
        return {k: (bool(v[b]) if v.dtype == bool else (None if not np.isfinite(v[b]) else float(v[b]))) for k, v in o.items()}

    def run(self, max_time=None, compact=True):
        """Run to the end. With compact=True (and no frame logging) finished rows are dropped from the batch once a
        quarter of it is done; their outcomes are frozen at that point (every outcome field stops changing when a row
        ends), so the result equals the uncompacted run row for row. Returns finish() in the original row order."""
        T = max_time or max(s["mission"]["max_time_s"] for s in self.sc)
        Tr = np.array([s["mission"]["max_time_s"] for s in self.sc])
        orig = np.arange(self.B)
        done = {}
        while self.t < T - 1e-9 and (self.phase < 4).any():
            self.step()
            over = (self.phase < 4) & (self.t >= Tr - 1e-9)
            self.t_end = np.where(over, self.t, self.t_end)
            self.end_pos = np.where(over[:, None], self.quad.pos[:, :2], self.end_pos)
            self.phase = np.where(over, 4, self.phase)
            fin = self.phase >= 4
            if compact and not self.log_rows and self.B >= 4 and fin.sum() >= max(2, self.B // 4) and (~fin).sum() >= 1:
                o = self.finish()
                for j in np.flatnonzero(fin):
                    done[orig[j]] = {k: v[j] for k, v in o.items()}
                keep = ~fin
                orig, Tr = orig[keep], Tr[keep]
                self._compact(keep)
        o = self.finish()
        for j in range(self.B):
            done[orig[j]] = {k: v[j] for k, v in o.items()}
        return {k: np.array([done[i][k] for i in range(len(done))]) for k in o}

    # Implementation note: per-row state is found by probing two tiny missions (B = 2 and 3) and keeping the attribute paths
    # whose leading dimension follows B; Dryden groups are re-indexed explicitly (their per-axis constants have length 3).
    _ROW_PATHS = {}

    def _row_paths(self):
        key = json.dumps([self.sc[0]["brain"]["arbiter"], self.sc[0]["brain"]["params"], _v2key(self.sc[0]["brain"])], sort_keys=True)
        if key not in Mission._ROW_PATHS:
            probes = []
            for n in (2, 3):
                m = Mission([dict(self.sc[0], seed=self.sc[0]["seed"])] * n)
                m.step()
                probes.append(dict(_walk(m)))
            a, b = probes
            Mission._ROW_PATHS[key] = [p for p in a if p in b and p[0] != "wind_groups" and _lead(a[p]) == 2 and _lead(b[p]) == 3]
        return Mission._ROW_PATHS[key]

    def _compact(self, keep):
        idx = np.flatnonzero(keep)
        for path in self._row_paths():
            v = _get(self, path)
            _set(self, path, v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
        new = np.full(len(keep), -1)
        new[idx] = np.arange(len(idx))
        groups = []
        for rows, d in self.wind_groups:
            k = keep[rows]
            if k.any():
                for name in ("mean", "x1", "x2", "wind"):
                    setattr(d, name, getattr(d, name)[k])
                d.rng.a = d.rng.a[k]
                d.B = int(k.sum())
                groups.append((new[rows[k]], d))
        self.wind_groups = groups
        for o in (self, self.S, self.quad, self.pi, self.ctrl, self.arb) + ((self.PS, self.hm) if self.v2 else ()) + \
                ((self.hd, self.sun_pi) if self.hd_on else ()):
            if hasattr(o, "B"):
                o.B = len(idx)
        self.last = None


def _walk(o, path=(), seen=None):
    """Yield (path, value) for every array / list / tuple reachable through attributes and dicts of a Mission."""
    seen = set() if seen is None else seen
    if id(o) in seen:
        return
    seen.add(id(o))
    items = o.items() if isinstance(o, dict) else vars(o).items()
    for k, v in items:
        p = path + (k,)
        if isinstance(v, (np.ndarray, list, tuple)):
            yield p, v
        elif isinstance(v, dict) or (hasattr(v, "__dict__") and not isinstance(v, (type, types.FunctionType, types.MethodType, types.ModuleType))):
            yield from _walk(v, p, seen)


def _lead(v):
    return (v.shape[0] if v.ndim else None) if isinstance(v, np.ndarray) else (len(v) if isinstance(v, list) else None)


def _get(o, path):
    for k in path:
        o = o[k] if isinstance(o, dict) else getattr(o, k)
    return o


def _set(o, path, v):
    parent = _get(o, path[:-1])
    if isinstance(parent, dict):
        parent[path[-1]] = v
    else:
        setattr(parent, path[-1], v)


def _v2key(br):
    return json.dumps([br.get("version", 1), br.get("modules"), br.get("module_params")], sort_keys=True)


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
        groups.setdefault(json.dumps([b["arbiter"], b["params"], b["ring_gain"], _v2key(b)], sort_keys=True), []).append(i)
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
                       "fault": meta.get("fault"), "severity": meta.get("severity"), "version": s["brain"].get("version", 1),
                       "modules": ",".join(s["brain"].get("modules", MODULES)) if s["brain"].get("version", 1) == 2 else ""}
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
    # v2 smoke: all modules, take-off learning flight stores pad views, v1 rows unaffected by the new fields
    m = Mission([sample_scenario(s, fault="none", arbiter="vote", landscape="forest", version=2, max_time_s=12.0) for s in (100, 101)], log_rows=[0])
    o = m.run()
    assert (m.hm.n_views > 30).all() and (m.phase >= 1).all() and set(o) >= {"landing_error_m", "landed_on_pad", "success_full"}
    assert {"hud", "home"} <= set(m.frames[0][-1]) and m.frames[0][-1]["hud"]["landing_stage"] == "cruise"
    try:
        Mission([{**sample_scenario(100, version=2), "target": {"kind": "pad", "pos": [30.0, 0.0, 0.0], "yaw_deg": 0.0}}])
        raise AssertionError("v2 accepted a pad target")
    except AssertionError as e:
        assert "3D objects" in str(e)
    # post-study adaptive_search: size estimate near the backpack's diagonal (1.005 m), altitude below Study 2's, HUD fields
    m = Mission([sample_scenario(100, fault="none", arbiter="vote", landscape="forest", version=2, max_time_s=12.0,
                                 modules=MODULES + ["adaptive_search"], target_kind="backpack")], log_rows=[0])
    o = m.run()
    assert abs(o["size_est_m"][0] - 1.005) < 0.15 and search.ALT_MIN <= o["search_alt_m"][0] < ALT_SEARCH
    assert {"search_alt_m", "size_est_m", "inspecting"} <= set(m.frames[0][-1]["hud"]) and m.f1[0] > m.f0[0]
    print("mission self-check OK (v1 + v2 smoke + adaptive_search)")
