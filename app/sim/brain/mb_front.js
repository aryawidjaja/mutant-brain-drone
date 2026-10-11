// Insect-style visual front end for the mushroom body (H8): JS twin of scripts/brain/mb_front.py (see its docstring).
// Receptor log compression -> luminance + two colour-opponent channels (RGB approximation of the bee's UV/blue/green;
// RGB has no UV) -> local contrast normalisation -> 4 oriented-edge channels (lobula-like, Roper et al. 2017) ->
// coarse retinotopic pooling -> stable rank / n. Python/JS agreement is asserted to <= 1e-9 on recorded crops.

export const EPS = 0.02, LCN_EPS = 1e-4;
export const CFG = { grid: 6, r: 2, colour_edges: true, per_channel: true, half_m: 1.1 };  // half_m: crop size used by the caller
const ANG = [0, 45, 90, 135].map((d) => d * Math.PI / 180);

function box(a, H, W, r) {          // (2r+1)^2 edge-clamped box mean, cumulative sums along x then y (as in Python)
  const tmp = new Float64Array(H * W), out = new Float64Array(H * W), n = 2 * r + 1;
  const pass = (src, dst, len, cnt, at) => {   // at(line, k) -> flat index
    const cs = new Float64Array(len + 2 * r + 1);
    for (let l = 0; l < cnt; l++) {
      let s = 0;
      for (let k = 0; k < len + 2 * r + 1; k++) {
        const kk = Math.min(Math.max(k - r - 1, 0), len - 1);
        s += src[at(l, kk)]; cs[k] = s;
      }
      for (let k = 0; k < len; k++) dst[at(l, k)] = (cs[k + n] - cs[k]) / n;
    }
  };
  pass(a, tmp, W, H, (l, k) => l * W + k);
  pass(tmp, out, H, W, (l, k) => k * W + l);
  return out;
}

function grad(a, H, W) {
  const gx = new Float64Array(H * W), gy = new Float64Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    gx[i] = (a[y * W + Math.min(x + 1, W - 1)] - a[y * W + Math.max(x - 1, 0)]) / 2;
    gy[i] = (a[Math.min(y + 1, H - 1) * W + x] - a[Math.max(y - 1, 0) * W + x]) / 2;
  }
  return [gx, gy];
}

/** rgb: H*W*3 (Uint8Array 0..255, or numbers 0..1) -> pooled raw features Float64Array(C * grid^2), channel-major. */
export function features(rgb, H, W, cfg = {}) {
  const c = { ...CFG, ...cfg }, N = H * W, sc = rgb instanceof Uint8Array ? 1 / 255 : 1;
  const lum = new Float64Array(N), obg = new Float64Array(N), org = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const r = Math.log(rgb[3 * i] * sc + EPS), g = Math.log(rgb[3 * i + 1] * sc + EPS), b = Math.log(rgb[3 * i + 2] * sc + EPS);
    lum[i] = (r + g + b) / 3; obg[i] = b - g; org[i] = r - g;
  }
  const m = box(lum, H, W, c.r), cc = new Float64Array(N), c2 = new Float64Array(N);
  for (let i = 0; i < N; i++) { cc[i] = lum[i] - m[i]; c2[i] = cc[i] * cc[i]; }
  const v = box(c2, H, W, c.r), nn = new Float64Array(N);
  for (let i = 0; i < N; i++) nn[i] = cc[i] / Math.sqrt(v[i] + LCN_EPS);
  const [gx, gy] = grad(nn, H, W);
  const ch = ANG.map((t) => { const e = new Float64Array(N), ct = Math.cos(t), st = Math.sin(t);
    for (let i = 0; i < N; i++) e[i] = Math.abs(ct * gx[i] + st * gy[i]); return e; });
  ch.push(obg, org);
  if (c.colour_edges) for (const o of [obg, org]) {
    const [ox, oy] = grad(o, H, W), e = new Float64Array(N);
    for (let i = 0; i < N; i++) e[i] = Math.sqrt(ox[i] * ox[i] + oy[i] * oy[i]);
    ch.push(e);
  }
  const G = c.grid, out = new Float64Array(ch.length * G * G);
  const ey = Array.from({ length: G + 1 }, (_, i) => Math.floor(H * i / G)), ex = Array.from({ length: G + 1 }, (_, i) => Math.floor(W * i / G));
  ch.forEach((a, k) => {
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      let s = 0;
      for (let y = ey[i]; y < ey[i + 1]; y++) for (let x = ex[j]; x < ex[j + 1]; x++) s += a[y * W + x];
      out[k * G * G + i * G + j] = s / ((ey[i + 1] - ey[i]) * (ex[j + 1] - ex[j]));
    }
  });
  return out;
}

function rank01(f, off, n, out) {  // stable rank of f[off..off+n) -> out[off..] = rank / n
  const o = Array.from({ length: n }, (_, i) => i).sort((a, b) => (f[off + a] - f[off + b]) || (a - b));
  for (let i = 0; i < n; i++) out[off + o[i]] = (i + 1) / n;
}

/** Object-centred crop -> MB input Float64Array(n_in) in (0, 1]. */
export function front(rgb, H, W, cfg = {}) {
  const c = { ...CFG, ...cfg }, f = features(rgb, H, W, c), x = new Float64Array(f.length);
  if (!c.per_channel) rank01(f, 0, f.length, x);
  else for (let off = 0, g2 = c.grid * c.grid; off < f.length; off += g2) rank01(f, off, g2, x);
  return x;
}

export const nIn = (cfg = {}) => { const c = { ...CFG, ...cfg }; return (c.colour_edges ? 8 : 6) * c.grid * c.grid; };

/** Inspection statistic: max over sample pairs >= sepDeg apart (view azimuth around the object, rad) of
 *  min(fam_i, fam_j); -Infinity if no valid pair. Stop allowed iff inspectStat(...) >= thr. */
export function inspectStat(viewAz, fam, valid, sepDeg = 20) {
  let best = -Infinity;
  const sep = sepDeg * Math.PI / 180 - 1e-12;
  for (let i = 0; i < fam.length; i++) for (let j = 0; j < fam.length; j++) {
    if (!valid[i] || !valid[j]) continue;
    let d = viewAz[i] - viewAz[j] + Math.PI;
    d = Math.abs(d - 2 * Math.PI * Math.floor(d / (2 * Math.PI)) - Math.PI);
    if (d >= sep) best = Math.max(best, Math.min(fam[i], fam[j]));
  }
  return best;
}

export const inspectOk = (viewAz, fam, valid, thr, sepDeg = 20) => inspectStat(viewAz, fam, valid, sepDeg) >= thr;
