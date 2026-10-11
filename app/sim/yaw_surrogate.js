// Surrogate of the Shiu-model yaw circuit (fitted transfer function, NOT the spiking model).
// H(s) = -K exp(-s tau) / (1 + s/wc), fitted in scripts/fit_yaw_surrogate.py -> results/yaw_surrogate.json.
// Input: yaw rate (deg/s). Output: DNa02 R-L rate difference (Hz). Discrete: delay line, then exact ZOH lag.
export class YawSurrogate {
  constructor(fit, dt = 0.002) {
    this.K = fit.K; this.n = Math.round(fit.tau_s / dt); this.a = Math.exp(-fit.wc_radps * dt);
    this.buf = new Array(this.n).fill(0); this.y = 0;
  }
  step(r) {
    this.buf.push(r);
    const u = this.buf.shift();
    this.y = this.a * this.y + (1.0 - this.a) * (-this.K * u);
    return this.y;
  }
}
