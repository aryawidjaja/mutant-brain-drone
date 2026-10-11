// Find-and-return mission, JS twin of scripts/mission.py (batch 1). Runs in a Web Worker or in Node; no DOM.
//   const m = createMission(scenario);  m.step();  m.frame()
// Protocol, constants and metric definitions: see scripts/mission.py (V2_DOC for brain.version 2).
// brain.version 1 (or absent) = the v1 brain bit for bit; 2 = the v2 brain with brain.modules (default all six:
// heading, home, landing, front, inspect, gated_arbiter). V2_BRAIN_DEFAULTS gives a ready v2 brain block.
// Frame (C6) extras: frame.home (v2: {stage, pad_visible, familiarity, locked, seen, searching, approach, views, touchdown})
// and frame.hud (both versions) for a drone-style HUD:
//   alt_m, alt_sp_m (altitude set-point), vz_mps, ground_speed_mps, air_speed_mps (|velocity - wind|),
//   heading_true_deg, heading_est_deg, heading_cmd_deg, roll_deg, pitch_deg, home_dist_est_m (|home vector|, v2: the
//   approach vector), home_dist_true_m, energy_so_far (N^1.5 s, the energy proxy so far), phase, version,
//   heading_conf (heading estimator confidence, v2 heading module else null), home_familiarity (v2 home module else
//   null), landing_stage ('cruise' | 'final' | 'descent' | 'touchdown', v2 landing module else null).
// v2 altitude: search and return at ALT_SEARCH; within INSPECT_NEAR m of a candidate (visible, strength > 0.3, while
// seeking) the set-point drops to max(ALT_INSPECT_MIN, object height); after the stop it climbs back; the final descent
// is the landing module's. brain.tip: absent = v1 instruction; {bearing_sd_deg, dist_sd}; null = no tip (spiral from home).
import { Rng } from './rng.js';
import { load, MODULES } from './scenario.js';
import { buildWorld } from './world.js';
import { Quad, Controller, P as QP } from './quad.js';
import { Dryden } from './dryden.js';
import * as eye from './brain/eye.js';
import { MushroomBody, ranks } from './brain/mb.js';
import { PathIntegrator, SIGMA_COMPASS, SIGMA_SPEED } from './brain/pi.js';
import { reflex, collided } from './brain/avoid.js';
import { say, steeringSlot } from './brain/mind.js';
import { make, driveInputs, N_RING } from './brain/arbiters.js';
import { YawSurrogate } from './yaw_surrogate.js';
import { HomeMemory, F_HOME, F_TRACK } from './brain/home.js';
import { features as mbFeatures, CFG as FRONT_CFG, inspectOk } from './brain/mb_front.js';
import { CLASSES as HEADING } from './brain/heading.js';
import { make as makeV2 } from './brain/arbiters_v2.js';
import { GAINS } from './vehicle_params.js';

export const DT_B = 0.02, SUB = 10, EYE_EVERY = 5, K_MAX = 4;
export const ALT = 2.5, V_CRUISE = 2.0, V_NEAR = 1.0, T_TAKEOFF = 2.0, TURN_MAX = 120 * Math.PI / 180;
export const STOP_R = 2.0, STOP_THR = 0.6, STOP_WIN = 10, F0 = 0.2, F1 = 0.6, SEARCH_S = 0.5, SPIRAL_B = 6.0;
const INSTR_BEAR_SD = 8.0 * Math.PI / 180, INSTR_DIST_SD = 0.1, HOLD_RAD = 20.0 * Math.PI / 180, CROSS_RATIO = 1.5, REL_CAP = 20.0;
const OCC_NEAR = 10.0, N_KC = 20000, K_IN = 10, KC_FRAC = 0.01, MB_SEED = 0;
const PHASES = ['learning', 'search', 'approach', 'home', 'done'];
// v2 brain (brain.version = 2): constants and protocol as scripts/mission.py V2_DOC
export const SPIRAL_B_NOTIP = 12.0, ALT_SEARCH = 3.5, ALT_INSPECT_MIN = 0.8, INSPECT_NEAR = 8.0;
export const HEADING_TUNED = { w_sun: 1.0868148932057646, gate_deg: 19.759606165836153, remap_delay_s: 1.573394669082153,
  remap_tau_s: 1.2535888902070214 };   // results/kt5_tuned_params.json estimators.ring (parity_brain.py asserts equality)
export const V2_BRAIN_DEFAULTS = { version: 2, modules: ['heading', 'home', 'landing', 'front', 'inspect', 'gated_arbiter'],
  module_params: { heading: HEADING_TUNED } };
export const T_HLF = 8.0, HLF_CLIMB = 1.0, R_FINAL = 6.0, D_SERVO = 2.0, K_SERVO = 0.8, V_FINAL = 0.6;
export const D_DESC = 0.25, V_HSTILL = 0.25, V_DESC = 0.3, V_DESC_H = 0.2, PAD_SEARCH_S = 40.0, PAD_SPIRAL_B = 3.0, PAD_R = 0.5;
const GYRO_BIAS_SD = 0.5 * Math.PI / 180, GYRO_NOISE_SD = 0.1 * Math.PI / 180;
export const HEADING_DEV = { w_sun: 0.5, gate_deg: 25.0, remap_delay_s: 3.0, remap_tau_s: 5.0 };
export const F0_FRONT = 0.330, F1_FRONT = 0.371, F_HOME_FRONT = 0.44, F_TRACK_FRONT = 0.22;
const INSPECT_WIN = 30, INSPECT_R = 4.0, INSPECT_SEP = 20.0, ORBIT_V = 0.5;
const V2_ARB = { ring: 'ring_gated', ring_static: 'ring_static_vg', vote: 'vote_vg', sum: 'sum_vg', accumulator: 'accumulator_vg',
  state_machine: 'state_machine' };
const wrap = eye.wrap;
const R2D = 180 / Math.PI;
const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

/** numpy's pairwise sum for a contiguous 1-D reduction of length n <= 128 (so the 1-s evidence mean is bit-identical). */
export function npSum(a) {
  const n = a.length;
  if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s; }
  const r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
  let i = 8;
  for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
  let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; i++) s += a[i];
  return s;
}

const MB_CACHE = {};
function freshMB(nIn = eye.LH * eye.LW) {
  if (!MB_CACHE[nIn]) MB_CACHE[nIn] = new MushroomBody(nIn, N_KC, K_IN, KC_FRAC, MB_SEED);
  const m = Object.create(MB_CACHE[nIn]);   // shares the wiring; own weights
  m.reset();
  return m;
}

/** opts.yawSurrogate = {fit: {K, tau_s, wc_radps}, k} adds an exploratory yaw-rate damping term (live app only; off by
 *  default, and off in every reported run): the fitted yaw-circuit surrogate turns the body yaw rate (deg/s) into a
 *  DNa02 rate difference (Hz), and k (rad/s per Hz) adds it to the yaw-rate set-point via the yaw set-point. */
export function createMission(scenario, opts = {}) {
  const sc = load(stripMeta(scenario));
  if (sc.decoys.length + 1 > K_MAX) throw new Error(`at most ${K_MAX - 1} decoys`);
  const world = buildWorld(sc);
  const S = new eye.Scene(world);
  const K = S.K;
  const home = sc.home.slice();
  const rComp = new Rng(sc.seed, 2), rSpeed = new Rng(sc.seed, 3), rEye = new Rng(sc.seed, 4), rSearch = new Rng(sc.seed, 6);
  const nb = rSearch.normal(), nd = rSearch.normal(), u0 = rSearch.uniform();
  const sw = sc.decoys.filter((d) => d.swap);
  const rep = sw.length ? sw[0].pos : sc.target.pos;
  const rel = [rep[0] - home[0], rep[1] - home[1]];
  const tip = 'tip' in sc.brain ? sc.brain.tip : 'v1';         // absent = the v1 instruction; null = no tip
  const bsd = tip === 'v1' || tip === null ? INSTR_BEAR_SD : tip.bearing_sd_deg * (Math.PI / 180);
  const dsd = tip === 'v1' || tip === null ? INSTR_DIST_SD : tip.dist_sd;
  const ang = Math.atan2(rel[1], rel[0]) + bsd * nb;
  const dist = Math.hypot(rel[0], rel[1]) * (1 + dsd * nd);
  const goal = tip === null ? [0.0, 0.0] : [dist * Math.cos(ang), dist * Math.sin(ang)];
  const spb = tip === null ? SPIRAL_B_NOTIP : SPIRAL_B;
  const yaw0 = 2 * Math.PI * u0 - Math.PI;
  const quad = new Quad(QP, [home[0] * 1, home[1] * 1, home[2] * 0], yaw0);
  const ctrl = new Controller();
  const ys = opts.yawSurrogate ? new YawSurrogate(opts.yawSurrogate.fit, DT_B / SUB) : null;
  const dryden = new Dryden(sc.seed, sc.wind.turbulence, sc.wind.mean_mps, ALT, V_CRUISE);
  const pi = new PathIntegrator();
  const v2 = (sc.brain.version ?? 1) === 2;
  if (v2 && (sc.target.kind === 'pad' || sc.decoys.some((d) => d.kind === 'pad')))
    throw new Error('brain.version 2 searches for 3D objects: no pad targets or pad decoys');
  const mods = new Set(v2 ? (sc.brain.modules ?? MODULES) : []);
  const mp = sc.brain.module_params ?? {};
  const useFront = mods.has('front'), hdOn = mods.has('heading');
  const lookOpts = useFront ? { halfM: FRONT_CFG.half_m, rgb: true } : {};
  const LH2 = eye.LH * eye.SS, LW2 = eye.LW * eye.SS, G2 = FRONT_CFG.grid * FRONT_CFG.grid;
  const LOOK_N = useFront ? LH2 * LW2 * 3 : eye.LH * eye.LW;
  const f0 = useFront ? F0_FRONT : F0, f1 = useFront ? F1_FRONT : F1;
  /** look -> MB input ranks (grey ranks, or the front end's integer per-channel ranks round(36 x)). */
  const code = (g) => {
    if (!useFront) return ranks(g);
    const f = mbFeatures(Float64Array.from(g, (v) => clip(v, 0.0, 1.0)), LH2, LW2);   // scripts/mission.py front_ranks
    const r = new Int32Array(f.length);
    for (let off = 0; off < f.length; off += G2) {
      const q = Array.from({ length: G2 }, (_, i) => Math.floor(f[off + i] * 1e9 + 0.5));
      const o = Array.from({ length: G2 }, (_, i) => i).sort((a, b) => (q[a] - q[b]) || (a - b));
      for (let i = 0; i < G2; i++) r[off + o[i]] = i + 1;
    }
    return r;
  };
  const mb = useFront ? freshMB(8 * G2) : freshMB();
  for (const g of eye.learningLooks(S, sc.brain.learning_flight_s, lookOpts)) mb.learn(code(g));
  const params = { ...sc.brain.params };
  if (sc.brain.arbiter === 'ring') params.ring_gain = sc.brain.ring_gain;
  const arb = mods.has('gated_arbiter') ? makeV2(V2_ARB[sc.brain.arbiter], params) : make(sc.brain.arbiter, params);
  const tTakeoff = mods.has('home') ? T_HLF : T_TAKEOFF;
  const fault = {};
  for (const f of sc.faults) fault[f.kind] = f;
  const falseSun = fault.false_sun ? (fault.false_sun.params.offset_deg ?? 0) * (Math.PI / 180) : 0.0;
  const tSun = fault.false_sun ? fault.false_sun.t_s : Infinity;
  const tComp = fault.compass_half ? fault.compass_half.t_s : Infinity;
  const speedScale = fault.speed_scale ? (fault.speed_scale.params.scale ?? 1.0) : 1.0;
  const tSpeed = fault.speed_scale ? fault.speed_scale.t_s : Infinity;
  const tCols = fault.eye_columns ? fault.eye_columns.t_s : Infinity;
  const deadCols = new Array(eye.NC).fill(false);
  if (fault.eye_columns) {
    const p = fault.eye_columns.params, n = Math.floor((p.frac ?? 0.25) * eye.NC + 0.5);
    for (let i = 0; i < n; i++) deadCols[((p.start_col ?? 0) + i) % eye.NC] = true;
  }
  const occDelay = fault.occlusion ? fault.occlusion.t_s : NaN;
  const occDur = fault.occlusion ? (fault.occlusion.params.duration_s ?? 5.0) : 0.0;
  const isT = S.role.map((r) => r === 1), isD = S.role.map((r) => r === 0);
  const hasDec = isD.some((x) => x);
  const tk = isT.indexOf(true);
  const Tmax = sc.mission.max_time_s;

  const st = {
    t: 0.0, tick: 0, phase: 0, psiSp: yaw0, headingEst: yaw0,
    vis: new Array(K).fill(false), bearRel: new Array(K).fill(0), lastBear: new Array(K).fill(0), rh: new Array(K).fill(Infinity),
    strength: new Array(K).fill(0), fam: new Array(K).fill(0), hist: Array.from({ length: K }, () => new Array(STOP_WIN).fill(0)),
    ev: new Array(K).fill(0), yawOff: 0, speedFac: 1, loom: 0, depth: new Float64Array(eye.NC).fill(eye.DEPTH_MAX),
    spiral: -1.0, spiral0: 0.0, out: { success_target: false, success_home: false, wrong_object_stop: false },
    collisions: 0, tEnd: NaN, endPos: null, path: 0, energy: 0, maxBank: 0, tFirstT: NaN, occOn: NaN, holdAlive: true, tHold: NaN,
    decSeen: false, tCross: NaN, tRel: NaN, wind: [0, 0, 0], snap: null, landed: false, landErr: NaN,
  };
  // v2 state
  let PS = null, rGyro = null, rPad = null, gyroBias = 0, gyroAcc = 0, hm = null, hd = null, sunPi = null;
  const objG = new Array(K).fill(null);
  const w2 = { padVis: false, homeFam: 0, seen: false, hstage: 0, ap: [0, 0], padT0: NaN, pspiral: 0, searching: false,
    altSp: ALT, touch: false, touchSpeed: NaN, touchPos: null, lfPhi0: wrap(yaw0 + Math.PI) };
  const ins = { az: [], ev: [], ok: [] };
  if (v2) {
    PS = new eye.Scene({ ...world, objects: [world.home_pad] });
    rGyro = new Rng(sc.seed, 20); rPad = new Rng(sc.seed, 21);
    gyroBias = GYRO_BIAS_SD * rGyro.normal();
    hm = new HomeMemory((g) => mb.active(code(g)), N_KC, useFront ? F_HOME_FRONT : F_HOME, useFront ? F_TRACK_FRONT : F_TRACK);
    if (hdOn) {
      const est = mp.heading_estimator ?? 'ring';
      hd = new HEADING[est](mp.heading ?? opts.headingParams ?? (est === 'ring' ? HEADING_TUNED : HEADING_DEV));
      sunPi = new PathIntegrator();
    }
    for (let k = 0; k < K; k++) { ins.az.push(new Array(INSPECT_WIN).fill(0)); ins.ev.push(new Array(INSPECT_WIN).fill(0)); ins.ok.push(new Array(INSPECT_WIN).fill(false)); }
  }

  const spiralWp = () => {
    const ph = Math.max(st.spiral, 0.0), r = spb * ph / (2 * Math.PI), a = st.spiral0 + ph;
    return [goal[0] + r * Math.cos(a), goal[1] + r * Math.sin(a)];
  };

  function eyeTick(pos, yaw, vmeas, dead, live) {
    const t = st.t;
    const occActive = t >= st.occOn && t < st.occOn + occDur;
    const occ = isT.map((x) => occActive && x);
    const v = eye.visibility(S, pos, yaw, t, dead, occ);
    const noise = eye.normals(rEye, K_MAX * LOOK_N);
    const fam = new Array(K).fill(0);
    for (let k = 0; k < K; k++) {
      if (!v.vis[k]) continue;
      const [grey] = eye.look(S, k, pos, t, lookOpts);
      const g = new Float64Array(grey.length);
      for (let i = 0; i < g.length; i++) g[i] = grey[i] + eye.NOISE * noise[k * LOOK_N + i];
      fam[k] = mb.familiarity(code(g));
      if (v2) objG[k] = g;
    }
    st.fam = fam; st.vis = v.vis; st.rh = v.rh;
    st.bearRel = v.bearing.map((b) => wrap(b - yaw));
    st.strength = fam.map((f, k) => (v.vis[k] ? clip((f - f0) / (f1 - f0), 0.0, 1.0) : 0.0));
    st.hist = st.hist.map((h, k) => [st.strength[k], ...h.slice(0, STOP_WIN - 1)]);
    st.ev = st.hist.map((h) => npSum(h) / STOP_WIN);
    if (mods.has('inspect')) for (let k = 0; k < K; k++) {   // view azimuth (object -> eye, compass frame), evidence, valid
      ins.az[k] = [wrap(st.bearRel[k] + st.headingEst + Math.PI), ...ins.az[k].slice(0, INSPECT_WIN - 1)];
      ins.ev[k] = [st.ev[k], ...ins.ev[k].slice(0, INSPECT_WIN - 1)];
      ins.ok[k] = [v.vis[k] && v.rh[k] < INSPECT_R, ...ins.ok[k].slice(0, INSPECT_WIN - 1)];
    }
    const tv = v.vis.some((x, k) => x && isT[k] && v.rh[k] < OCC_NEAR) && (st.phase === 1 || st.phase === 2);
    if (Number.isNaN(st.tFirstT) && tv) st.tFirstT = t;
    if (Number.isNaN(st.occOn) && !Number.isNaN(occDelay)) st.occOn = st.tFirstT + occDelay;
    let evT = 0, evD = 0;
    for (let k = 0; k < K; k++) { evT += st.ev[k] * (isT[k] ? 1 : 0); evD += st.ev[k] * (isD[k] ? 1 : 0); }
    st.decSeen = st.decSeen || evD > 0;
    if (Number.isNaN(st.tCross) && st.decSeen && evT > CROSS_RATIO * evD && live && st.phase < 3) st.tCross = t;
    const fn = eye.normals(rEye, eye.NC);
    st.depth = eye.depthRing(S, pos, yaw);
    const r = reflex(st.depth, vmeas, fn, dead);
    st.yawOff = r.yawOffset; st.speedFac = r.speedFactor; st.loom = r.loom;
  }

  function eyeHome(pos, yaw, dead, hv) {
    const t = st.t;
    const lf = st.phase === 0 && t >= HLF_CLIMB;
    const hom = st.phase === 3 && (Math.hypot(hv[0], hv[1]) < R_FINAL || hm.locked);
    const pz = eye.visibility(PS, pos, yaw, t, dead, null);
    const pv = pz.vis[0] && (lf || hom);
    const noise = eye.normals(rPad, LOOK_N);
    const g = new Float64Array(LOOK_N);
    if (pv) { const [gl] = eye.look(PS, 0, pos, t, lookOpts); for (let i = 0; i < LOOK_N; i++) g[i] = gl[i] + eye.NOISE * noise[i]; }
    w2.padVis = pv;
    if (pv && lf) hm.learn(g);
    const r = hm.steer([g, ...objG], [pv && hom, ...st.vis.map((x) => x && hom)],
      [wrap((pz.bearing[0] - yaw) + st.headingEst), ...st.bearRel.map((b) => wrap(b + st.headingEst))], [pz.rh[0], ...st.rh], hv);
    w2.homeFam = r.fam; w2.seen = r.seen;
  }

  const padWp = () => { const r = PAD_SPIRAL_B * w2.pspiral / (2 * Math.PI); return [r * Math.cos(w2.pspiral), r * Math.sin(w2.pspiral)]; };

  /** approach vector (compass frame): visual once locked, else PI; spiral point during the pad search. -> bearing */
  function approach(hv) {
    let ap = mods.has('home') && hm.locked ? [hv[0] + hm.corr[0], hv[1] + hm.corr[1]] : [hv[0], hv[1]];
    if (w2.searching) {
      for (let it = 0; it < 100; it++) {
        const p = padWp(), wp = [p[0] + hv[0], p[1] + hv[1]];
        if (!(Math.hypot(wp[0], wp[1]) < 1.5)) break;
        const r = PAD_SPIRAL_B * w2.pspiral / (2 * Math.PI);
        w2.pspiral = w2.pspiral + Math.min(0.5, 1.0 / Math.max(r, 1.0));
      }
      const p = padWp();
      ap = [p[0] + hv[0], p[1] + hv[1]];
    }
    w2.ap = ap;
    return Math.atan2(ap[1], ap[0]);
  }

  function landing(pos) {
    const home_ = st.phase === 3, d = Math.hypot(w2.ap[0], w2.ap[1]);
    if (!mods.has('landing')) return home_ && d < 1.0;
    const locked = hm.locked && mods.has('home');
    if (home_ && w2.hstage === 0 && (d < R_FINAL || locked)) w2.hstage = 1;
    if (mods.has('home')) {
      if (home_ && w2.hstage === 1 && !locked && d < 1.0 && Number.isNaN(w2.padT0)) w2.padT0 = st.t;
      w2.searching = home_ && w2.hstage === 1 && !locked && st.t - (Number.isNaN(w2.padT0) ? Infinity : w2.padT0) < PAD_SEARCH_S;
    }
    const hs = Math.hypot(quad.vel[0], quad.vel[1]);
    if (home_ && w2.hstage === 1 && !w2.searching && d < D_DESC && hs < V_HSTILL) { w2.altSp = pos[2]; w2.hstage = 2; }
    return home_ && w2.touch;
  }

  function spV2(sp, yaw, hv, live) {
    let vel = sp.vel_xy.slice(), alt = sp.alt, yw = sp.yaw;
    // altitude: search/return high, drop to the object's height near a candidate (inspection)
    let kc = -1;
    for (let k = 0; k < K; k++) {
      const c = st.vis[k] && st.rh[k] < INSPECT_NEAR && st.strength[k] > 0.3 && (st.phase === 1 || st.phase === 2);
      if (c && (kc < 0 || st.rh[k] < st.rh[kc])) kc = k;
    }
    alt = live ? (kc >= 0 ? Math.max(ALT_INSPECT_MIN, S.obj[kc][6]) : ALT_SEARCH) : 0.0;
    const rot = yaw - st.headingEst, c = Math.cos(rot), s = Math.sin(rot);
    const toAir = (q) => [c * q[0] - s * q[1], s * q[0] + c * q[1]];
    if (mods.has('home') && st.phase === 0) {
      const tau = st.t;
      if (tau < HLF_CLIMB) alt = 1.0; else {
        const u = (tau - HLF_CLIMB) / (T_HLF - HLF_CLIMB);
        const r = 0.5 + 1.7 * u, dr = 1.7 / (T_HLF - HLF_CLIMB);
        const ph = w2.lfPhi0 + 1.5 * Math.PI * u, dph = 1.5 * Math.PI / (T_HLF - HLF_CLIMB);
        const e = [Math.cos(ph), Math.sin(ph)];
        let vc = [dr * e[0] + r * dph * -e[1] + 1.5 * (r * e[0] + hv[0]), dr * e[1] + r * dph * e[0] + 1.5 * (r * e[1] + hv[1])];
        const f = Math.min(1.0, 2.0 / Math.max(Math.hypot(vc[0], vc[1]), 1e-9));
        vc = [vc[0] * f, vc[1] * f];
        vel = toAir(vc); alt = 1.0 + (ALT_SEARCH - 1.0) * u; yw = wrap(Math.atan2(vc[1], vc[0]) + rot);
      }
    }
    if (mods.has('inspect') && (st.phase === 1 || st.phase === 2) && st.vis.some((x, k) => x && st.rh[k] < INSPECT_R && st.ev[k] > 0.3))
      vel = [vel[0] + ORBIT_V * -Math.sin(yw), vel[1] + ORBIT_V * Math.cos(yw)];
    if (mods.has('landing')) {
      const d = Math.hypot(w2.ap[0], w2.ap[1]);
      const servo = st.phase === 3 && ((w2.hstage === 1 && d < D_SERVO && !w2.searching) || w2.hstage === 2);
      if (servo) {
        let vc = [K_SERVO * w2.ap[0], K_SERVO * w2.ap[1]];
        const f = Math.min(1.0, (w2.hstage === 2 ? V_DESC_H : V_FINAL) / Math.max(Math.hypot(vc[0], vc[1]), 1e-9));
        vc = [vc[0] * f, vc[1] * f];
        vel = toAir(vc);
        yw = d > 0.5 ? wrap(Math.atan2(w2.ap[1], w2.ap[0]) + rot) : yaw;
      }
      if (w2.hstage === 2 && live) { w2.altSp = Math.max(w2.altSp - V_DESC * DT_B, -0.2); alt = w2.altSp; }
    }
    return { vel_xy: vel, alt, yaw: yw };
  }

  function finish() {
    const tEnd = Number.isNaN(st.tEnd) ? st.t : st.tEnd;
    const started = !Number.isNaN(st.occOn) && occDur > 0;
    const tHold = started && Number.isNaN(st.tHold) && st.occOn < tEnd ? Math.min(occDur, tEnd - st.occOn) : st.tHold;
    const tRel = !Number.isNaN(st.tCross) && Number.isNaN(st.tRel) ? Math.min(REL_CAP, tEnd - st.tCross) : Math.min(st.tRel, REL_CAP);
    const num = (x) => (Number.isFinite(x) ? x : null);
    return { ...st.out, time_s: tEnd, path_len_m: st.path, collisions: st.collisions,
      home_error_m: Math.hypot((st.endPos ?? quad.pos)[0] - home[0], (st.endPos ?? quad.pos)[1] - home[1]), max_bank_deg: st.maxBank, energy_proxy: st.energy,
      t_hold_occlusion_s: occDur > 0 ? num(tHold) : null, t_release_decoy_s: hasDec ? num(tRel) : null,
      landing_error_m: num(st.landErr), landed_on_pad: st.landed && st.landErr <= PAD_R,
      touchdown_speed_mps: v2 ? num(w2.touchSpeed) : null,
      success_full: st.out.success_target && st.landed && st.landErr <= PAD_R && st.collisions === 0 };
  }

  function step() {
    const t = st.t;
    const yaw = quad.euler()[2];
    const pos = Float64Array.from(quad.pos);
    let live = st.phase < 4;
    const sunOn = t >= tSun, colsOn = t >= tCols;
    for (let i = 0; i < 16; i++) pi.colMask[i] = !hdOn && t >= tComp && (i % 8) < 4 ? 0.0 : 1.0;
    const theta = yaw + SIGMA_COMPASS * rComp.normal() + (sunOn ? falseSun : 0.0);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const vel = quad.vel;
    const vb = [c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]];
    const n1 = rSpeed.normal(), n2 = rSpeed.normal();
    const scale = t >= tSpeed ? speedScale : 1.0;
    const vmeas = [scale * vb[0] + SIGMA_SPEED * n1, scale * vb[1] + SIGMA_SPEED * n2];
    let thetaPi = theta;
    if (hdOn) {      // heading module: gyro (stream 20) + the v1 sun compass (compass_half = its CL1 mask) -> fused heading
      const gyro = gyroAcc / SUB + gyroBias + GYRO_NOISE_SD * rGyro.normal();
      for (let i = 0; i < 16; i++) sunPi.colMask[i] = t >= tComp && (i % 8) < 4 ? 0.0 : 1.0;
      const sun = sunPi.step(theta, [0, 0], DT_B);
      thetaPi = hd.step(gyro, sun, true, DT_B);
    }
    st.headingEst = pi.step(thetaPi, vmeas, DT_B);
    const hv = pi.homeVector();
    const dead = colsOn ? deadCols : null;
    if (st.tick % EYE_EVERY === 0) {
      eyeTick(pos, yaw, vmeas, dead, live);
      if (mods.has('home')) eyeHome(pos, yaw, dead, hv);
    }
    const bearing = st.bearRel.map((b, k) => (st.vis[k] ? wrap(b + st.headingEst) : st.lastBear[k]));   // hidden: last seen
    st.lastBear = bearing;
    // drives
    const pHat = [-hv[0], -hv[1]];
    const toGoal = [goal[0] - pHat[0], goal[1] - pHat[1]];
    if (st.spiral < 0 && Math.hypot(toGoal[0], toGoal[1]) < 3.0) { st.spiral0 = Math.atan2(-toGoal[1], -toGoal[0]); st.spiral = 0.0; }
    for (let it = 0; it < 100; it++) {
      const wp = spiralWp();
      if (!(st.spiral >= 0 && Math.hypot(wp[0] - pHat[0], wp[1] - pHat[1]) < 3.0)) break;
      const r = spb * st.spiral / (2 * Math.PI);
      st.spiral = st.spiral + Math.min(0.5, 1.0 / Math.max(r, 1.0));
    }
    const tg = st.spiral >= 0 ? spiralWp() : goal;
    const searchB = Math.atan2(tg[1] - pHat[1], tg[0] - pHat[0]);
    const homing = st.phase === 3;
    let seeking = st.phase === 1 || st.phase === 2;
    const obs = {
      objects: { bearing, strength: st.strength.map((x) => (seeking ? x : 0.0)), visible: st.vis.map((x) => x && seeking) },
      home: { bearing: v2 ? approach(hv) : Math.atan2(hv[1], hv[0]), strength: homing ? 1.0 : 0.0 },
      search: { bearing: searchB, strength: seeking || st.phase === 0 ? SEARCH_S : 0.0 },
      heading: st.headingEst,
    };
    const res = arb.step(obs, DT_B);
    const cmd = res.heading_cmd;
    // metrics (KT2 definitions)
    const tb = S.obj.map((o) => wrap(Math.atan2(o[1] - pos[1], o[0] - pos[0]) - yaw + st.headingEst));
    const onT = Math.abs(wrap(cmd - tb[tk])) <= HOLD_RAD;
    const occ = t >= st.occOn && t < st.occOn + occDur;
    if (occ && st.holdAlive && !onT && live) { st.tHold = t - st.occOn; st.holdAlive = false; }
    if (!Number.isNaN(st.tCross) && Number.isNaN(st.tRel) && onT && live) st.tRel = t - st.tCross;
    // phases, shared stop rule, landing
    if (st.phase === 0 && t >= tTakeoff) st.phase = 1;
    seeking = st.phase === 1 || st.phase === 2;
    const onObj = st.vis.some((v, k) => v && st.strength[k] > 0.2 && Math.abs(wrap(bearing[k] - cmd)) < 30 * Math.PI / 180);
    if (seeking) st.phase = res.commit > 0.5 && onObj ? 2 : 1;
    let stT = false, stD = false;
    for (let k = 0; k < K; k++) {
      let cand = st.vis[k] && st.rh[k] < STOP_R && st.ev[k] > STOP_THR && seeking;
      if (cand && mods.has('inspect')) cand = inspectOk(ins.az[k], ins.ev[k], ins.ok[k], STOP_THR, INSPECT_SEP);
      if (cand && isT[k]) stT = true;
      if (cand && isD[k]) stD = true;
    }
    stD = stD && !stT;
    st.out.success_target = st.out.success_target || stT;
    st.out.wrong_object_stop = st.out.wrong_object_stop || stD;
    if (stT || stD) st.phase = 3;
    const landed = v2 ? landing(pos) : st.phase === 3 && Math.hypot(hv[0], hv[1]) < 1.0;
    const hdTrue = Math.hypot(pos[0] - home[0], pos[1] - home[1]);
    if (landed && live && !st.landed) { st.landErr = hdTrue; st.landed = true; }
    st.out.success_home = st.out.success_home || (landed && st.out.success_target && hdTrue < STOP_R);
    const crash = live && collided(S, pos);
    st.collisions += crash ? 1 : 0;
    if (live && (landed || crash)) { st.tEnd = t; st.phase = 4; st.endPos = [pos[0], pos[1]]; }
    // descending-neuron level
    live = st.phase < 4;
    const want = cmd + st.yawOff;
    st.psiSp = wrap(st.psiSp + clip(wrap(want - st.psiSp), -TURN_MAX * DT_B, TURN_MAX * DT_B));
    const yawSp = wrap(st.psiSp - st.headingEst + yaw);
    const nearObj = st.vis.some((v, k) => v && st.rh[k] < 5.0 && st.strength[k] > 0.3);
    let v = V_CRUISE * st.speedFac;
    if (nearObj && seeking) v = Math.min(v, V_NEAR);
    if (st.phase === 3) v = Math.min(v, Math.max(0.4, (v2 ? Math.hypot(w2.ap[0], w2.ap[1]) : Math.hypot(hv[0], hv[1])) / 2));
    if (st.phase === 0 || !live) v = 0.0;
    let sp = { vel_xy: [v * Math.cos(yawSp), v * Math.sin(yawSp)], alt: live ? ALT : 0.0, yaw: st.phase === 0 ? yaw : yawSp };
    if (v2) sp = spV2(sp, yaw, hv, live);
    st.snap = { t, tick: st.tick, yaw, pos, vel: Float64Array.from(quad.vel), quat: Float64Array.from(quad.quat),
      omega: Float64Array.from(quad.omega), motor: Float64Array.from(quad.motor), euler: quad.euler(), wind: st.wind.slice(),
      hv, res, obs, dead, sunOn, phase: st.phase, vis: st.vis.slice(), strength: st.strength.slice(), rh: st.rh.slice(),
      fam: st.fam.slice(), headingEst: st.headingEst, yawOff: st.yawOff, loom: st.loom,
      altSp: sp.alt, energy: st.energy, hdConf: hd ? hd.conf : null,
      home2: v2 ? { stage: w2.hstage, pad_visible: w2.padVis, familiarity: w2.homeFam, locked: hm.locked, seen: w2.seen,
        searching: w2.searching, approach: w2.ap.slice(), views: hm.nViews, touchdown: w2.touch } : null };
    const dt = DT_B / SUB;
    if (v2) gyroAcc = 0;
    for (let i = 0; i < SUB; i++) {
      st.wind = Array.from(dryden.step(dt));
      let m = ctrl.update(quad, ys ? { ...sp, yaw: sp.yaw + opts.yawSurrogate.k * ys.step(quad.omega[2] * R2D) / GAINS.kp_yaw } : sp, dt);
      if (v2 && w2.touch) m = [0, 0, 0, 0];                 // disarmed after touchdown
      const vprev = v2 ? Array.from(quad.vel) : null;
      quad.step(m, st.wind, dt);
      if (v2) {
        gyroAcc += quad.omega[2];
        if (w2.hstage === 2 && !w2.touch && quad.pos[2] <= 0.0 && st.phase < 4) {
          w2.touchSpeed = Math.sqrt(vprev[0] * vprev[0] + vprev[1] * vprev[1] + vprev[2] * vprev[2]);
          w2.touchPos = [quad.pos[0], quad.pos[1]]; w2.touch = true;
        }
      }
      const lv = st.phase < 4 ? 1 : 0;
      st.path += lv * Math.hypot(quad.vel[0], quad.vel[1]) * DT_B / SUB;
      let e = 0;
      for (let j = 0; j < 4; j++) e += Math.max(quad.motor[j], 0) ** 1.5;
      st.energy += lv * e * DT_B / SUB;
      const q = quad.quat;
      const tilt = Math.acos(clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)) * R2D;
      if (lv) st.maxBank = Math.max(st.maxBank, tilt);
    }
    st.tick += 1;
    st.t = st.tick * DT_B;
    if (st.phase < 4 && st.t >= Tmax - 1e-9) { st.tEnd = st.t; st.phase = 4; st.endPos = [quad.pos[0], quad.pos[1]]; }
  }

  /** C6 frame for the most recent brain tick (state at the start of that tick, as scripts/mission.py logs it). */
  function frame() {
    const z = st.snap;
    if (!z) return null;
    const { res, obs } = z;
    let act, drv;
    if (res.viz.ring && res.viz.drives) { act = Array.from(res.viz.ring); drv = res.viz.drives; } else {
      drv = driveInputs(obs, 2.0, 1.0, 1.0, 1.0);
      act = new Array(N_RING).fill(0);
    }
    const visB = z.vis;
    let fam = 0, anyVis = false;
    for (let k = 0; k < K; k++) if (visB[k]) { fam = anyVis ? Math.max(fam, z.fam[k]) : z.fam[k]; anyVis = true; }
    const slot = steeringSlot(res.heading_cmd, obs.objects.bearing, z.strength, visB);
    const roles = S.role.map((r) => (r ? 'target' : 'decoy'));
    const fa = Object.keys(fault).filter((k) => z.t >= fault[k].t_s && k !== 'occlusion');
    if (z.t >= st.occOn && z.t < st.occOn + occDur) fa.push('occlusion');
    const homeTrue = [home[0] - z.pos[0], home[1] - z.pos[1]];
    const m = say(PHASES[z.phase], z.headingEst, z.hv, fam, slot, z.yaw, homeTrue, roles, z.rh, fa);
    const rgb = eye.panorama(S, z.pos, z.yaw, z.t, z.dead, z.sunOn && falseSun !== 0 ? S.sunAz + falseSun : null);
    const viz = {};
    for (const [k, v] of Object.entries(res.viz)) if (k !== 'ring' && k !== 'drives') viz[k] = plain(v);
    return {
      t: Math.round(z.t * 1e6) / 1e6, phase: PHASES[z.phase], outcome: z.phase === 4 ? finish() : null,
      quad: { pos: Array.from(z.pos), vel: Array.from(z.vel), quat: Array.from(z.quat), euler_deg: z.euler.map((x) => x * R2D),
        omega: Array.from(z.omega), motors: Array.from(z.motor, (f) => f / QP.f_max) },
      wind: z.wind,
      eye: { w: eye.NC, h: eye.NR, rgb },
      familiarity: fam,
      objects: Array.from({ length: K }, (_, k) => ({ slot: k, bearing_deg: obs.objects.bearing[k] * R2D, strength: z.strength[k], visible: visB[k] })),
      ring: { n: N_RING, activity: act, drives: { objects: Array.from(drv.objects), home: Array.from(drv.home), search: Array.from(drv.search) },
        heading_cmd_deg: res.heading_cmd * R2D },
      arbiter: { name: sc.brain.arbiter, commit: res.commit, viz },
      pi: { home_est: z.hv.slice(), home_true: homeTrue, heading_est_deg: z.headingEst * R2D },
      avoid: { yaw_offset_deg: z.yawOff * R2D, loom: z.loom },
      mind: { text: m.text, truth: m.truth },
      faults_active: fa,
      ...(z.home2 ? { home: z.home2 } : {}),
      hud: { alt_m: z.pos[2], alt_sp_m: z.altSp, vz_mps: z.vel[2], ground_speed_mps: Math.hypot(z.vel[0], z.vel[1]),
        air_speed_mps: Math.hypot(z.vel[0] - z.wind[0], z.vel[1] - z.wind[1], z.vel[2] - z.wind[2]),
        heading_true_deg: z.yaw * R2D, heading_est_deg: z.headingEst * R2D, heading_cmd_deg: res.heading_cmd * R2D,
        roll_deg: z.euler[0] * R2D, pitch_deg: z.euler[1] * R2D,
        home_dist_est_m: v2 ? Math.hypot(z.home2.approach[0], z.home2.approach[1]) : Math.hypot(z.hv[0], z.hv[1]),
        home_dist_true_m: Math.hypot(homeTrue[0], homeTrue[1]), energy_so_far: z.energy, phase: PHASES[z.phase],
        version: v2 ? 2 : 1, heading_conf: z.hdConf, home_familiarity: mods.has('home') ? z.home2.familiarity : null,
        landing_stage: mods.has('landing') ? (z.home2.touchdown ? 'touchdown' : ['cruise', 'final', 'descent'][z.home2.stage]) : null },
    };
  }

  return {
    scenario: sc, world, step, frame, finish,
    get t() { return st.t; }, get phase() { return PHASES[st.phase]; }, get done() { return st.phase === 4; },
  };
}

function plain(v) {
  if (ArrayBuffer.isView(v)) return Array.from(v);
  if (Array.isArray(v)) return v.map(plain);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)]));
  return v;
}

function stripMeta(s) {
  const o = {};
  for (const k of Object.keys(s)) if (!k.startsWith('_')) o[k] = s[k];
  return o;
}

/** Run to the end (Node parity runner, "Run 20 seeds" button). frameEvery = N brain ticks between frames (0 = none). */
export function runMission(scenario, { maxTime = null, frameEvery = 0 } = {}) {
  const m = createMission(scenario);
  const T = maxTime ?? m.scenario.mission.max_time_s;
  const frames = [];
  while (m.t < T - 1e-9 && !m.done) {
    const tick = Math.round(m.t / DT_B);
    m.step();
    if (frameEvery && tick % frameEvery === 0) frames.push(m.frame());
  }
  return { outcome: m.finish(), frames };
}
