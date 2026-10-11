// The six heading arbiters (contract C2), JS port of scripts/brain/arbiters.py and scripts/brain/state_machine.py
// (both owned by workstream A; mirrored line by line, batch 1). obs: {objects: {bearing[K], strength[K], visible[K]},
// home: {bearing, strength}, search: {bearing, strength}, heading}. step(obs, dt) -> {heading_cmd, commit, viz}.
export const ARBITERS = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine'];
export const N_RING = 32;
const SIGMA_DEG = 30.0, J_E = 24.0, C_INH = 8.0;
export const TH = Array.from({ length: N_RING }, (_, i) => i * 2 * Math.PI / N_RING);
const pmod = (x, m) => { const r = x % m; return (r !== 0 && (r < 0) !== (m < 0)) ? r + m : r; };
export const wrap = (a) => pmod(a + Math.PI, 2 * Math.PI) - Math.PI;

const W_RING = (() => {
  const G = TH.map((a) => TH.map((b) => { const d = Math.abs(wrap(a - b)) * (180 / Math.PI); return Math.exp(-(d ** 2) / (2 * SIGMA_DEG ** 2)); }));
  let m = 0; for (let j = 0; j < N_RING; j++) m += G[0][j];
  const JI = (m / N_RING) * J_E + C_INH;
  return G.map((row) => row.map((g) => (J_E * g - JI) / N_RING));
})();

const vm = (kappa, mu) => TH.map((t) => Math.exp(kappa * (Math.cos(t - mu) - 1.0)));

/** Shared input stage: von Mises bumps scaled by gain * strength. -> {objects, home, search} arrays [N] */
export function driveInputs(obs, kappa, gObj, gHome, gSearch) {
  const o = obs.objects, objects = new Float64Array(N_RING);
  for (let k = 0; k < o.bearing.length; k++) {
    const s = o.strength[k] * (o.visible[k] ? 1 : 0), v = vm(kappa, o.bearing[k]);
    for (let i = 0; i < N_RING; i++) objects[i] += s * v[i];
  }
  for (let i = 0; i < N_RING; i++) objects[i] = gObj * objects[i];
  const hv = vm(kappa, obs.home.bearing), sv = vm(kappa, obs.search.bearing);
  const gh = gHome * obs.home.strength, gs = gSearch * obs.search.strength;
  return { objects, home: Float64Array.from(hv, (v) => gh * v), search: Float64Array.from(sv, (v) => gs * v) };
}

function pv(r, heading) {
  let x = 0, y = 0;
  for (let i = 0; i < N_RING; i++) { x += r[i] * Math.cos(TH[i]); y += r[i] * Math.sin(TH[i]); }
  return Math.hypot(x, y) > 1e-6 ? Math.atan2(y, x) : heading;
}

const check = (p, D, name) => { for (const k of Object.keys(p)) if (!(k in D)) throw new Error(`unknown ${name} param ${k}`); return { ...D, ...p }; };

export class Ring {
  static DEFAULTS = { kappa: 2.0, g_obj: 1.5, g_home: 1.5, g_search: 0.5, tau_s: 0.2, ring_gain: 1.0 };
  constructor(params = {}, name = 'ring') {
    this.name = name;
    this.p = check(params, this.constructor.DEFAULTS, name);
    this.r = new Float64Array(N_RING);
    const g = this.p.ring_gain ?? 0.0;
    this.W = W_RING.map((row) => row.map((w) => g * w));
  }
  step(obs, dt) {
    const p = this.p, d = driveInputs(obs, p.kappa, p.g_obj, p.g_home, p.g_search);
    const I = Float64Array.from(d.objects, (v, i) => v + d.home[i] + d.search[i]);
    const g = p.ring_gain ?? 0.0;
    const n = Math.max(1, Math.ceil(dt / p.tau_s * (1.0 + g * C_INH)));
    const h = dt / p.tau_s / n;
    for (let it = 0; it < n; it++) {
      const nr = new Float64Array(N_RING);
      for (let j = 0; j < N_RING; j++) {
        let s = 0;
        for (let i = 0; i < N_RING; i++) s += this.r[i] * this.W[i][j];
        nr[j] = this.r[j] + h * (Math.min(Math.max(s + I[j], 0.0), 1.0) - this.r[j]);
      }
      this.r = nr;
    }
    return { heading_cmd: wrap(pv(this.r, obs.heading)), commit: Math.max(...this.r), viz: { ring: Float64Array.from(this.r), drives: d } };
  }
}

export class RingStatic extends Ring {
  static DEFAULTS = { kappa: 2.0, g_obj: 1.5, g_home: 1.5, g_search: 0.5, tau_s: 0.2 };
  constructor(params = {}) { super(params, 'ring_static'); }
}

export class Vote {
  static DEFAULTS = { kappa: 2.0, g_obj: 1.5, g_home: 1.5, g_search: 0.5 };
  constructor(params = {}) { this.name = 'vote'; this.p = check(params, Vote.DEFAULTS, 'vote'); }
  step(obs, dt) {
    const p = this.p, d = driveInputs(obs, p.kappa, p.g_obj, p.g_home, p.g_search);
    const I = Float64Array.from(d.objects, (v, i) => v + d.home[i] + d.search[i]);
    let i = 0;
    for (let j = 1; j < N_RING; j++) if (I[j] > I[i]) i = j;
    const ym = I[(i - 1 + N_RING) % N_RING], y0 = I[i], yp = I[(i + 1) % N_RING];
    const den = ym - 2 * y0 + yp;
    const off = Math.abs(den) > 1e-12 ? 0.5 * (ym - yp) / den : 0.0;
    const cmd = TH[i] + Math.min(Math.max(off, -0.5), 0.5) * 2 * Math.PI / N_RING;
    return { heading_cmd: wrap(y0 > 1e-9 ? cmd : obs.heading), commit: Math.min(Math.max(y0, 0), 1), viz: { ring: I, drives: d } };
  }
}

export class Sum {
  static DEFAULTS = { g_obj: 1.5, g_home: 1.5, g_search: 0.5, p_obj: 1.0 };
  constructor(params = {}) { this.name = 'sum'; this.p = check(params, Sum.DEFAULTS, 'sum'); }
  step(obs, dt) {
    const p = this.p, o = obs.objects;
    let ox = 0, oy = 0;
    for (let k = 0; k < o.bearing.length; k++) {
      const s = (o.strength[k] * (o.visible[k] ? 1 : 0)) ** p.p_obj;
      ox += s * Math.cos(o.bearing[k]); oy += s * Math.sin(o.bearing[k]);
    }
    const vec = {
      objects: [p.g_obj * ox, p.g_obj * oy],
      home: [p.g_home * obs.home.strength * Math.cos(obs.home.bearing), p.g_home * obs.home.strength * Math.sin(obs.home.bearing)],
      search: [p.g_search * obs.search.strength * Math.cos(obs.search.bearing), p.g_search * obs.search.strength * Math.sin(obs.search.bearing)],
    };
    const zx = vec.objects[0] + vec.home[0] + vec.search[0], zy = vec.objects[1] + vec.home[1] + vec.search[1];
    const a = Math.hypot(zx, zy);
    return { heading_cmd: wrap(a > 1e-9 ? Math.atan2(zy, zx) : obs.heading), commit: Math.min(Math.max(a, 0), 1), viz: { vectors: vec } };
  }
}

export class Accumulator {
  static DEFAULTS = { g_obj: 1.5, g_home: 1.5, g_search: 0.5, tau_leak_s: 1.0 };
  constructor(params = {}) { this.name = 'accumulator'; this.p = check(params, Accumulator.DEFAULTS, 'accumulator'); this.a = null; }
  step(obs, dt) {
    const p = this.p, o = obs.objects, K = o.bearing.length;
    if (this.a === null) { this.a = new Float64Array(K + 2); this.last = Float64Array.from(o.bearing); }
    for (let k = 0; k < K; k++) if (o.visible[k]) this.last[k] = o.bearing[k];
    const e = [];
    for (let k = 0; k < K; k++) e.push(p.g_obj * o.strength[k] * (o.visible[k] ? 1 : 0));
    e.push(p.g_home * obs.home.strength, p.g_search * obs.search.strength);
    const al = 1.0 - Math.exp(-dt / p.tau_leak_s);
    for (let i = 0; i < K + 2; i++) this.a[i] += al * (e[i] - this.a[i]);
    const bearings = [...this.last, obs.home.bearing, obs.search.bearing];
    let k = 0, tot = 0;
    for (let i = 0; i < K + 2; i++) { if (this.a[i] > this.a[k]) k = i; tot += this.a[i]; }
    const amax = this.a[k];
    return { heading_cmd: wrap(amax > 1e-9 ? bearings[k] : obs.heading), commit: amax / Math.max(tot, 1e-9),
      viz: { evidence: Float64Array.from(this.a), choice: k } };
  }
}

// ---------------------------------------------------------------- state machine (blind-authored, frozen; mirror)
const SEARCH = 0, APPROACH = 1, HOLD = 2, HOME = 3;
export const SM_STATES = ['SEARCH', 'APPROACH', 'HOLD', 'HOME'];
export class StateMachine {
  static DEFAULTS = { ev_tau_s: 0.3, bearing_tau_s: 0.1, enter_thr: 0.5, exit_ratio: 0.7, debounce_s: 0.2, debounce_frac: 0.7,
    switch_margin: 0.25, drop_frac: 0.6, timeout_s: 2.0, reject_s: 5.0 };
  constructor(params = {}) { this.name = 'state_machine'; this.p = check(params, StateMachine.DEFAULTS, 'state_machine'); this.K = null; }
  _init(K, dt) {
    const p = this.p;
    this.K = K;
    this.W = Math.max(1, pyRoundInt(p.debounce_s / dt));
    this.kReq = Math.max(1, Math.ceil(p.debounce_frac * this.W));
    this.exitThr = p.exit_ratio * p.enter_thr;
    this.t = 0;
    this.state = SEARCH; this.target = -1;
    this.ev = new Float64Array(K); this.bvec = Array.from({ length: K }, () => [0, 0]);
    this.seen = new Array(K).fill(false); this.hiddenT = new Float64Array(K); this.rejectT = new Float64Array(K);
    this.peak = 0;
    this.hAcq = Array.from({ length: K }, () => new Array(this.W).fill(false));
    this.hSw = Array.from({ length: K }, () => new Array(this.W).fill(false));
    this.hBad = new Array(this.W).fill(false);
  }
  _commit(slot) {
    this.target = slot; this.state = APPROACH; this.peak = this.ev[slot];
    this.hSw = this.hSw.map((r) => r.fill(false)); this.hBad.fill(false);
  }
  step(obs, dt) {
    const o = obs.objects, K = o.bearing.length;
    if (this.K === null) this._init(K, dt);
    const p = this.p, slot = this.t % this.W;
    this.t += 1;
    const aEv = 1.0 - Math.exp(-dt / p.ev_tau_s), aB = 1.0 - Math.exp(-dt / p.bearing_tau_s);
    const fbear = new Float64Array(K), ok = [];
    for (let k = 0; k < K; k++) {
      const vis = !!o.visible[k], first = vis && !this.seen[k];
      const u = [Math.cos(o.bearing[k]), Math.sin(o.bearing[k])];
      if (vis) {
        this.ev[k] = first ? o.strength[k] : this.ev[k] + aEv * (o.strength[k] - this.ev[k]);
        this.bvec[k] = first ? u : [this.bvec[k][0] + aB * (u[0] - this.bvec[k][0]), this.bvec[k][1] + aB * (u[1] - this.bvec[k][1])];
      }
      this.seen[k] = this.seen[k] || vis;
      fbear[k] = Math.atan2(this.bvec[k][1], this.bvec[k][0]);
      this.hiddenT[k] = vis ? 0.0 : this.hiddenT[k] + dt;
      this.rejectT[k] = Math.max(this.rejectT[k] - dt, 0.0);
      ok.push(vis && this.rejectT[k] <= 0);
    }
    let hasT = this.target >= 0, tgt = hasT ? this.target : 0;
    let evT = this.ev[tgt];
    const visT = !!o.visible[tgt] && hasT;
    if (visT) this.peak = Math.max(this.peak, evT);
    for (let k = 0; k < K; k++) {
      const notT = k !== this.target;
      this.hAcq[k][slot] = ok[k] && this.ev[k] > p.enter_thr;
      this.hSw[k][slot] = ok[k] && notT && hasT && this.ev[k] > p.enter_thr && this.ev[k] > (1.0 + p.switch_margin) * evT;
    }
    this.hBad[slot] = visT && (evT < this.exitThr || evT < p.drop_frac * this.peak);
    const cnt = (a) => a.reduce((s, x) => s + (x ? 1 : 0), 0);
    const acq = this.hAcq.map((h) => cnt(h) >= this.kReq);
    const sw = this.hSw.map((h) => cnt(h) >= this.kReq);
    const bad = cnt(this.hBad) >= this.kReq;
    const homing = obs.home.strength > 0;
    if (!homing && this.state === HOME) this.state = SEARCH;
    const active = !homing && hasT;
    if (active && bad) { this.rejectT[tgt] = p.reject_s; this.target = -1; this.state = SEARCH; }
    for (let k = 0; k < K; k++) { acq[k] = acq[k] && this.rejectT[k] <= 0; sw[k] = sw[k] && this.rejectT[k] <= 0; }
    const s = active && !bad && sw.some((x) => x);
    if (s) {
      let best = -1;
      for (let k = 0; k < K; k++) if (sw[k] && (best < 0 || this.ev[k] > this.ev[best])) best = k;
      this._commit(best);
    }
    const keep = active && !bad && !s;
    if (keep && visT) this.state = APPROACH;
    const hold = keep && !visT;
    if (hold) this.state = HOLD;
    if (hold && this.hiddenT[tgt] > p.timeout_s) { this.target = -1; this.state = SEARCH; }
    if (!homing && this.target < 0 && acq.some((x) => x)) {
      let best = -1;
      for (let k = 0; k < K; k++) if (acq[k] && (best < 0 || this.ev[k] > this.ev[best])) best = k;
      this._commit(best);
    }
    if (homing) this.state = HOME;
    tgt = this.target >= 0 ? this.target : 0;
    const searchH = obs.search.strength > 0 ? obs.search.bearing : obs.heading;
    const cmd = this.state === HOME ? obs.home.bearing : (this.state === APPROACH || this.state === HOLD) ? fbear[tgt] : searchH;
    evT = this.ev[tgt];
    const band = Math.max(p.enter_thr - this.exitThr, 1e-9);
    const c01 = (x) => Math.min(Math.max(x, 0), 1);
    const commit = this.state === HOME ? 1.0 : this.state === APPROACH ? c01((evT - this.exitThr) / band)
      : this.state === HOLD ? c01((evT - this.exitThr) / band) * c01(1 - this.hiddenT[tgt] / p.timeout_s) : 0.0;
    return { heading_cmd: wrap(cmd), commit,
      viz: { state: this.state, target: this.target, evidence: Float64Array.from(this.ev), rejected: Array.from(this.rejectT, (x) => x > 0) } };
  }
}

/** Python int(round(x)) (half to even) for the debounce window. */
function pyRoundInt(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const CLASSES = { ring: Ring, ring_static: RingStatic, vote: Vote, sum: Sum, accumulator: Accumulator, state_machine: StateMachine };
export function make(name, params = {}) {
  if (!(name in CLASSES)) throw new Error(`unknown arbiter ${name}`);
  return new CLASSES[name]({ ...params });
}
export const DEFAULTS = Object.fromEntries(Object.entries(CLASSES).map(([k, c]) => [k, c.DEFAULTS]));
