// v2 arbiters, JS port of scripts/brain/arbiters_v2.py (batch 1): the evidence-gated ring (ring_gated) and
// visibility-gated versions of the v1 arms (*_vg). Same obs / step(obs, dt) interface as arbiters.js.
import { Ring, RingStatic, Vote, Sum, Accumulator, make as makeV1, driveInputs, wrap, TH, N_RING, DEFAULTS as DEFAULTS_V1 } from './arbiters.js';

const SIGMA_DEG = 30.0, J_E = 24.0, C_INH = 8.0;  // same constants as arbiters.js
const W_RING = (() => {
  const G = TH.map((a) => TH.map((b) => { const d = Math.abs(wrap(a - b)) * (180 / Math.PI); return Math.exp(-(d ** 2) / (2 * SIGMA_DEG ** 2)); }));
  let m = 0; for (let j = 0; j < N_RING; j++) m += G[0][j];
  const JI = (m / N_RING) * J_E + C_INH;
  return G.map((row) => row.map((g) => (J_E * g - JI) / N_RING));
})();
export const G_CRIT = 0.22877692472876562;      // 1 / lambda_max(W_RING), asserted equal to Python's value in its self-check
export const K_C = Math.log(2.0) / (1.0 - Math.cos(Math.PI / 4));
const X_W = 0.1;
export const ARBITERS_V2 = ['ring_gated', 'ring_vg', 'ring_static_vg', 'vote_vg', 'sum_vg', 'accumulator_vg', 'state_machine'];

/** Visibility-gated input: hidden slots keep their last-seen bearing and a strength decaying at tau_hold_s. */
class Hold {
  constructor(tauHold) { this.tau = tauHold; this.hs = null; }
  apply(obs, dt) {
    const o = obs.objects, K = o.bearing.length;
    if (this.hs === null) { this.hs = new Float64Array(K); this.hb = Float64Array.from(o.bearing); }
    const dec = Math.exp(-dt / this.tau);
    for (let k = 0; k < K; k++) {
      if (o.visible[k]) { this.hs[k] = o.strength[k]; this.hb[k] = o.bearing[k]; } else this.hs[k] = this.hs[k] * dec;
    }
    return { ...obs, objects: { bearing: Array.from(this.hb), strength: Array.from(this.hs), visible: Array.from(this.hs, (s) => s > 0) } };
  }
}

const split = (params, name) => {
  const { tau_hold_s: tau = 2.0, ...rest } = params;
  return [tau, rest];
};

class VisGated {
  constructor(Base, name, params) {
    const [tau, rest] = split(params, name);
    this.name = name; this.inner = new Base(rest); this.hold = new Hold(tau);
    this.p = { ...this.inner.p, tau_hold_s: tau };
  }
  step(obs, dt) { return this.inner.step(this.hold.apply(obs, dt), dt); }
}

export class RingGated {
  static DEFAULTS = { ...Ring.DEFAULTS, lo_frac: 0.5, tau_m_s: 0.5, theta_c: 0.2, tau_hold_s: 2.0 };
  constructor(params = {}) {
    for (const k of Object.keys(params)) if (!(k in RingGated.DEFAULTS)) throw new Error(`unknown ring_gated param ${k}`);
    this.name = 'ring_gated';
    this.p = { ...RingGated.DEFAULTS, ...params };
    this.r = new Float64Array(N_RING);
    this.m = 0;
    this.gHi = this.p.ring_gain;
    this.gLo = Math.min(this.gHi, this.p.lo_frac * G_CRIT);
    this.hold = new Hold(this.p.tau_hold_s);
  }
  step(obs, dt) {
    const p = this.p, d = driveInputs(this.hold.apply(obs, dt), p.kappa, p.g_obj, p.g_home, p.g_search);
    const I = Float64Array.from(d.objects, (v, i) => v + d.home[i] + d.search[i]);
    let zx = 0, zy = 0;
    for (let i = 0; i < N_RING; i++) { zx += this.r[i] * Math.cos(TH[i]); zy += this.r[i] * Math.sin(TH[i]); }
    const thb = Math.atan2(zy, zx);
    let sc = -Infinity, sa = -Infinity;
    for (let i = 0; i < N_RING; i++) {
      const k = Math.exp(K_C * (Math.cos(TH[i] - thb) - 1.0));
      sc = Math.max(sc, I[i] * k); sa = Math.max(sa, I[i] * (1.0 - k));
    }
    const x = (sa - sc) / Math.max(sa + sc, 1e-9);
    const c = Math.hypot(zx, zy) > 1e-6 ? Math.min(Math.max((x - p.theta_c) / X_W, 0.0), 1.0) : 0.0;
    this.m += (1.0 - Math.exp(-dt / p.tau_m_s)) * (c - this.m);
    const g = this.gHi + (this.gLo - this.gHi) * this.m;
    const n = Math.max(1, Math.ceil(dt / p.tau_s * (1.0 + this.gHi * C_INH)));
    const h = dt / p.tau_s / n;
    for (let it = 0; it < n; it++) {
      const nr = new Float64Array(N_RING);
      for (let j = 0; j < N_RING; j++) {
        let s = 0;
        for (let i = 0; i < N_RING; i++) s += this.r[i] * W_RING[i][j];
        nr[j] = this.r[j] + h * (Math.min(Math.max(s * g + I[j], 0.0), 1.0) - this.r[j]);
      }
      this.r = nr;
    }
    let px = 0, py = 0;
    for (let i = 0; i < N_RING; i++) { px += this.r[i] * Math.cos(TH[i]); py += this.r[i] * Math.sin(TH[i]); }
    const heading = Math.hypot(px, py) > 1e-6 ? Math.atan2(py, px) : obs.heading;
    return { heading_cmd: wrap(heading), commit: Math.max(...this.r),
      viz: { ring: Float64Array.from(this.r), drives: d, gain: g, conflict: this.m } };
  }
}

const VG = { ring_vg: Ring, ring_static_vg: RingStatic, vote_vg: Vote, sum_vg: Sum, accumulator_vg: Accumulator };
export function make(name, params = {}) {
  if (name === 'ring_gated') return new RingGated({ ...params });
  if (name in VG) return new VisGated(VG[name], name, { ...params });
  return makeV1(name, params);
}
export const DEFAULTS = { ...DEFAULTS_V1, ring_gated: RingGated.DEFAULTS,
  ...Object.fromEntries(Object.entries(VG).map(([k, c]) => [k, { ...c.DEFAULTS, tau_hold_s: 2.0 }])) };
