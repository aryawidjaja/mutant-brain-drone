// Mushroom-body familiarity (Ardin et al. 2016 style), JS port of scripts/brain/mb.py as used by the mission.
// Wiring: idx[j][i] = floor(u * nIn) for the uniforms of new Rng(seed, 10) drawn in order j = 0..nKc-1, i = 0..kIn-1.
// KC active if its drive >= the k-th largest drive; familiarity = 1 - |active AND w| / |active|; 1 bit per KC->MBON.
// The mission feeds integer pixel ranks (rank equalisation of mb.equalise, whose L2 normalisation is a common factor),
// so KC drives are exact integers in both languages (scripts/mission.py ranks()/kc_active()).
import { Rng } from '../rng.js';

export const MB_STREAM = 10;

export function wiring(nIn, nKc, kIn, seed) {
  const r = new Rng(seed, MB_STREAM);
  const idx = new Int32Array(nKc * kIn);
  for (let n = 0; n < nKc * kIn; n++) idx[n] = Math.floor(r.uniform() * nIn);
  return idx;
}

/** grey values -> ranks 1..n after quantising to 1/4096; ties broken by pixel index (stable sort). */
export function ranks(grey) {
  const n = grey.length, q = new Float64Array(n);
  for (let i = 0; i < n; i++) q[i] = Math.floor(grey[i] * 4096 + 0.5);
  const o = Array.from({ length: n }, (_, i) => i).sort((a, b) => (q[a] - q[b]) || (a - b));
  const r = new Int32Array(n);
  for (let i = 0; i < n; i++) r[o[i]] = i + 1;
  return r;
}

export class MushroomBody {
  constructor(nIn, nKc = 20000, kIn = 10, kcActiveFrac = 0.01, seed = 0) {
    this.nIn = nIn; this.nKc = nKc; this.kIn = kIn;
    this.k = Math.max(1, Math.round(kcActiveFrac * nKc));
    this.idx = wiring(nIn, nKc, kIn, seed);
    this.reset();
  }
  reset() { this.w = new Uint8Array(this.nKc).fill(1); }
  /** rank vector (Int32Array nIn) -> Uint8Array(nKc) active flags */
  active(rk) {
    const d = new Int32Array(this.nKc), kIn = this.kIn, idx = this.idx;
    for (let j = 0; j < this.nKc; j++) { let s = 0; for (let i = 0; i < kIn; i++) s += rk[idx[j * kIn + i]]; d[j] = s; }
    const thr = Int32Array.from(d).sort()[this.nKc - this.k];
    const a = new Uint8Array(this.nKc);
    for (let j = 0; j < this.nKc; j++) a[j] = d[j] >= thr ? 1 : 0;
    return a;
  }
  learn(rk) { const a = this.active(rk); for (let j = 0; j < this.nKc; j++) if (a[j]) this.w[j] = 0; }
  familiarity(rk) {
    const a = this.active(rk);
    let n = 0, m = 0;
    for (let j = 0; j < this.nKc; j++) if (a[j]) { n++; if (this.w[j]) m++; }
    return 1.0 - m / n;
  }
  get nbytes() { return (this.nKc + 7) >> 3; }
}
