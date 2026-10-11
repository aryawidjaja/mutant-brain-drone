// MIL-F-8785C low-altitude Dryden turbulence. Line-by-line port of scripts/dryden.py (batch 1); see that file.
import { Rng } from './rng.js';

const FT = 0.3048, KT = 0.514444, SQ3 = Math.sqrt(3);
export const W20_KT = { none: 0, light: 15, moderate: 30, severe: 45 };

export function drydenScales(intensity, altM) {
  const h = Math.max(altM / FT, 10.0);
  const k = 0.177 + 0.000823 * h;
  const Lu = h / k ** 1.2 * FT;
  const sw = 0.1 * W20_KT[intensity] * KT;
  const su = sw / k ** 0.4;
  return { sigma: [su, su, sw], L: [Lu, Lu, h * FT] };
}

export class Dryden {
  constructor(seed, intensity = 'none', mean = [0, 0, 0], altM = 3.0, airspeed = 2.0) {
    if (!(intensity in W20_KT)) throw new Error(`unknown turbulence intensity: ${intensity}`);
    this.intensity = intensity; this.V = airspeed; this.mean = Float64Array.from(mean);
    const { sigma, L } = drydenScales(intensity, altM);
    this.sigma = sigma; this.L = L;
    this.T = L.map(l => l / this.V);
    this.gain = sigma.map((s, i) => s * Math.sqrt((i === 0 ? 2.0 : 1.0) * L[i] / (Math.PI * this.V)));
    this.rng = new Rng(seed, 1);
    this.x1 = new Float64Array(3); this.x2 = new Float64Array(3);
    this.wind = Float64Array.from(mean);
  }
  step(dt) {
    if (this.intensity === 'none') return Float64Array.from(this.mean);
    const s = Math.sqrt(Math.PI / dt);
    const n = [this.rng.normal() * s, this.rng.normal() * s, this.rng.normal() * s];
    const w = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      const a = Math.exp(-dt / this.T[i]);
      this.x1[i] = a * this.x1[i] + (1.0 - a) * n[i];
      this.x2[i] = a * this.x2[i] + (1.0 - a) * this.x1[i];
      const turb = i === 0 ? this.gain[0] * this.x1[0] : this.gain[i] * (SQ3 * this.x1[i] + (1.0 - SQ3) * this.x2[i]);
      w[i] = this.mean[i] + turb;
    }
    this.wind = w;
    return Float64Array.from(w);
  }
}
