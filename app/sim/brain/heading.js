// Heading estimators fusing a yaw-rate gyro with the sun compass: JS twin (batch 1) of scripts/brain/heading.py.
// See that file for the model. est = new HeadingRing(params); heading = est.step(gyroZ, sunTheta, sunValid, dt); est.conf
export const N_RING = 32;
const TWO_PI = 2 * Math.PI, DEG = Math.PI / 180, TAU_CONF = 1.0;
export const PHI = Array.from({ length: N_RING }, (_, i) => TWO_PI * i / N_RING);
const D_RING = TWO_PI / N_RING;
const COS_PHI = PHI.map(Math.cos), SIN_PHI = PHI.map(Math.sin);
const W_REC = PHI.map((a) => PHI.map((b) => (2.0 / N_RING) * Math.cos(a - b)));
const P0_PSI = (5 * DEG) ** 2, P0_BIAS = (0.5 * DEG) ** 2, REMAP_END = 0.2;

export const wrap = (a) => a - TWO_PI * Math.floor((a + Math.PI) / TWO_PI);
const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

class Base {
  constructor(params) {
    this.p = {};
    for (const k of Object.keys(this.constructor.PARAM_SPACE)) this.p[k] = params[k];
    this.init = false;
    this.conf = 0;
  }
  _conf(acc, dt) { this.conf = this.conf + dt / TAU_CONF * ((acc ? 1 : 0) - this.conf); }
}

export class HeadingRing extends Base {
  static name_ = 'ring';
  static PARAM_SPACE = { w_sun: [0.05, 5.0, 'log'], gate_deg: [5.0, 120.0, 'lin'],
    remap_delay_s: [0.5, 30.0, 'log'], remap_tau_s: [1.0, 1000.0, 'log'] };
  constructor(params) {
    super(params);
    this.e = Float64Array.from(COS_PHI, (c) => 1 + c);
    this.o = 0;
    this.timer = 0;
    this.remap = false;
  }
  decode() {
    let y = 0, x = 0;
    for (let i = 0; i < N_RING; i++) { y += this.e[i] * SIN_PHI[i]; x += this.e[i] * COS_PHI[i]; }
    return Math.atan2(y, x);
  }
  step(gyroZ, sunTheta, sunValid, dt) {
    const x = Math.min(Math.abs(gyroZ) * dt, D_RING);
    const s = Math.sin(x) / (Math.sin(x) + Math.sin(D_RING - x));
    const e0 = this.e, e = new Float64Array(N_RING);
    for (let i = 0; i < N_RING; i++) {
      const r = gyroZ >= 0 ? e0[(i + N_RING - 1) % N_RING] : e0[(i + 1) % N_RING];   // PEN shift
      e[i] = (1 - s) * e0[i] + s * r;
    }
    this.e = e;
    if (sunValid && !this.init) {
      const cs = Math.cos(sunTheta), sn = Math.sin(sunTheta);
      for (let i = 0; i < N_RING; i++) this.e[i] = 1.0 + COS_PHI[i] * cs + SIN_PHI[i] * sn;
      this.o = 0.0;
      this.remap = false;
    }
    if (sunValid) this.init = true;
    const psi = this.decode();
    const c = Math.abs(wrap(sunTheta - this.o - psi));
    const gate = this.p.gate_deg * DEG;
    const agree = sunValid && c <= gate;
    this.timer = agree ? 0.0 : (sunValid ? this.timer + dt : this.timer);
    const done = this.remap && sunValid && c <= REMAP_END * gate;
    if (done) this.o = wrap(sunTheta - psi);
    this.remap = (this.remap || (sunValid && this.timer > this.p.remap_delay_s)) && !done;
    const acc = agree && !this.remap;
    if (this.remap && sunValid)
      this.o = wrap(this.o + dt / this.p.remap_tau_s * wrap(sunTheta - psi - this.o));
    const w = (acc ? 1 : 0) * this.p.w_sun * dt, cc = Math.cos(sunTheta - this.o), cs = Math.sin(sunTheta - this.o);
    const u = new Float64Array(N_RING);
    for (let i = 0; i < N_RING; i++) u[i] = this.e[i] + w * (1.0 + COS_PHI[i] * cc + SIN_PHI[i] * cs);
    const a = new Float64Array(N_RING);
    let ss = 0;
    for (let j = 0; j < N_RING; j++) {
      let v = 0;
      for (let i = 0; i < N_RING; i++) v += u[i] * W_REC[i][j];
      a[j] = v; ss += v * v;
    }
    const amp = Math.sqrt(2.0 * (ss / N_RING));
    for (let i = 0; i < N_RING; i++) this.e[i] = 1.0 + a[i] / amp;
    this._conf(acc, dt);
    return this.decode();
  }
}

export class ComplementaryFilter extends Base {
  static name_ = 'cf';
  static PARAM_SPACE = { k_sun: [0.05, 5.0, 'log'], clamp_deg: [1.0, 90.0, 'log'], reject_deg: [5.0, 180.0, 'lin'] };
  constructor(params) { super(params); this.psi = 0; }
  step(gyroZ, sunTheta, sunValid, dt) {
    let psi = wrap(this.psi + gyroZ * dt);
    if (sunValid && !this.init) psi = sunTheta;
    if (sunValid) this.init = true;
    const y = wrap(sunTheta - psi);
    const acc = sunValid && Math.abs(y) <= this.p.reject_deg * DEG;
    const cl = this.p.clamp_deg * DEG;
    this.psi = acc ? wrap(psi + this.p.k_sun * dt * clip(y, -cl, cl)) : psi;
    this._conf(acc, dt);
    return this.psi;
  }
}

export class KalmanHeading extends Base {
  static name_ = 'kf';
  static PARAM_SPACE = { q_psi: [0.01, 2.0, 'log'], q_bias: [1e-4, 0.1, 'log'], r_sun_deg: [0.5, 20.0, 'log'], chi2: [1.0, 100.0, 'log'] };
  constructor(params) {
    super(params);
    this.psi = 0; this.b = 0; this.p00 = P0_PSI; this.p01 = 0; this.p11 = P0_BIAS;
  }
  step(gyroZ, sunTheta, sunValid, dt) {
    const qp = (this.p.q_psi * DEG) ** 2, qb = (this.p.q_bias * DEG) ** 2;
    let psi = wrap(this.psi + (gyroZ - this.b) * dt);
    let p00 = this.p00 - 2 * dt * this.p01 + dt * dt * this.p11 + qp * dt;
    let p01 = this.p01 - dt * this.p11;
    let p11 = this.p11 + qb * dt;
    let b = this.b;
    if (sunValid && !this.init) { psi = sunTheta; b = 0.0; p00 = P0_PSI; p01 = 0.0; p11 = P0_BIAS; }
    if (sunValid) this.init = true;
    const y = wrap(sunTheta - psi);
    const S = p00 + (this.p.r_sun_deg * DEG) ** 2;
    const acc = sunValid && y * y / S <= this.p.chi2;
    const k0 = p00 / S, k1 = p01 / S;
    if (acc) {
      this.psi = wrap(psi + k0 * y); this.b = b + k1 * y;
      this.p00 = p00 - k0 * p00; this.p11 = p11 - k1 * p01; this.p01 = p01 - k0 * p01;
    } else {
      this.psi = psi; this.b = b; this.p00 = p00; this.p01 = p01; this.p11 = p11;
    }
    this._conf(acc, dt);
    return this.psi;
  }
}

export const CLASSES = { ring: HeadingRing, cf: ComplementaryFilter, kf: KalmanHeading };
