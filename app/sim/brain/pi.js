// Central-complex path integrator (Stone et al. 2017, reimplemented from the paper's text), JS twin of
// scripts/brain/pi.py. See that file for the equations and the three stated deviations.
export const N8 = 8;
export const TH8 = Array.from({ length: N8 }, (_, i) => i * Math.PI / 4);
export const C_TB1 = 0.33, PHI_TN = Math.PI / 4, V_REF = 4.0, H_CPU4 = 0.025;
export const SIGMA_COMPASS = 2.0 * (Math.PI / 180), SIGMA_SPEED = 0.05;
const A_ = { TL: 3.0, CL1: 3.0, TB1: 5.0 }, B_ = { TL: 0.0, CL1: -1.5, TB1: 1.0 };
export const GAIN = -109.81636878389264;   // real part of pi.py GAIN (imaginary part is 0 to 1e-13)
const W_TB1 = TH8.map((a) => TH8.map((b) => (Math.cos(a - b) - 1) / 2));
const sig = (x, l) => 1.0 / (1.0 + Math.exp(-(A_[l] * x - B_[l])));

export class PathIntegrator {
  constructor() {
    this.tb1 = new Float64Array(N8);
    this.cpu4 = [new Float64Array(N8).fill(0.5), new Float64Array(N8).fill(0.5)];   // [L, R][direction]
    this.colMask = new Float64Array(2 * N8).fill(1.0);
  }
  /** thetaH compass estimate (rad), vBody [forward, left] m/s -> heading estimate (rad) */
  step(thetaH, vBody, dt) {
    const rcl1 = new Float64Array(2 * N8);
    for (let i = 0; i < 2 * N8; i++) {
      const rtl = sig(Math.cos(TH8[i % N8] - thetaH), 'TL');
      rcl1[i] = sig(-rtl, 'CL1') * this.colMask[i];
    }
    const prev = this.tb1, tb1 = new Float64Array(N8);
    for (let j = 0; j < N8; j++) {
      let s = 0;
      for (let i = 0; i < N8; i++) s += (C_TB1 * prev[i]) * W_TB1[i][j];   // (c * r_TB1) @ W, as numpy parses it
      tb1[j] = sig((1 - C_TB1) * (0.5 * (rcl1[j] + rcl1[j + N8])) + s, 'TB1');
    }
    this.tb1 = tb1;
    const [f, l] = vBody;
    const tn = [Math.cos(PHI_TN) * f + Math.sin(PHI_TN) * l, Math.cos(PHI_TN) * f - Math.sin(PHI_TN) * l];
    const rtn = tn.map((x) => Math.min(Math.max(x / V_REF, 0.0), 1.0));
    let mean = 0;
    for (let i = 0; i < N8; i++) mean += 1 - tb1[i];
    mean /= N8;
    for (let h = 0; h < 2; h++)
      for (let i = 0; i < N8; i++)
        this.cpu4[h][i] = Math.min(Math.max(this.cpu4[h][i] + H_CPU4 * dt * rtn[h] * ((1 - tb1[i]) - mean), 0.0), 1.0);
    return this.heading();
  }
  heading() {
    let x = 0, y = 0;
    for (let i = 0; i < N8; i++) { x += this.tb1[i] * Math.cos(TH8[i]); y += this.tb1[i] * Math.sin(TH8[i]); }
    return Math.atan2(-y, -x);
  }
  homeVector() {
    let x = 0, y = 0;
    for (let h = 0; h < 2; h++)
      for (let i = 0; i < N8; i++) {
        const a = TH8[i] + (h === 0 ? PHI_TN : -PHI_TN);
        x += this.cpu4[h][i] * Math.cos(a); y += this.cpu4[h][i] * Math.sin(a);
      }
    return [GAIN * x, GAIN * y];
  }
}
